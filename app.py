#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diet_analyzer.py
----------------
Streamlit 기반 식단 분석 & 다음 식사 제안 도구
- 각 식사 슬롯별 음식 및 컨디션 입력
- food_db.csv 기반 영양 태그 분석
- 부족한 영양소 자동 감지 후 다음 식사 추천
- log.csv 자동 저장 및 최신 입력 복원 (무한복사 방지)
"""

from __future__ import annotations
import re, sys, ast, json, base64, zlib
from collections import defaultdict
from typing import List, Dict, Tuple, Any
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None

# ====================== 설정 ======================
FOOD_DB_CSV = "food_db.csv"
NUTRIENT_DICT_CSV = "nutrient_dict.csv"
LOG_CSV = "log.csv"
FOOD_DB_UPDATED_CSV = "food_db_updated.csv"

SLOTS = [
    "아침", "아침보조제", "오전 간식", "점심", "점심보조제",
    "오후 간식", "저녁", "저녁보조제", "저녁 간식"
]
TZ = ZoneInfo("Europe/Paris")

# ==================== 날짜/상태 관리 ====================
def today_str() -> str:
    return datetime.now(TZ).date().isoformat()

def next_midnight():
    now = datetime.now(TZ)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=TZ)

def init_daily_state():
    """자정 기준 하루 단위 상태 초기화 + log.csv에서 최신 1건 복원"""
    if "daily_date" not in st.session_state:
        st.session_state.daily_date = today_str()
    if st.session_state.daily_date != today_str():
        for k in ["inputs", "conditions", "last_items_df", "last_nutri_df", "last_recs", "last_combo"]:
            st.session_state.pop(k, None)
        st.session_state.daily_date = today_str()

    st.session_state.setdefault("inputs", {s: "" for s in SLOTS})
    st.session_state.setdefault("conditions", {s: "" for s in SLOTS})
    st.session_state.setdefault("last_items_df", None)
    st.session_state.setdefault("last_nutri_df", None)
    st.session_state.setdefault("last_recs", [])
    st.session_state.setdefault("last_combo", [])
    st.session_state.setdefault("threshold", 1)
    st.session_state.setdefault("export_flag", True)

    # 오늘 날짜 로그 복원
    try:
        df_log = pd.read_csv(LOG_CSV)
        today_logs = df_log[df_log["date"] == today_str()]
        if not today_logs.empty:
            for slot in SLOTS:
                slot_logs = today_logs[today_logs["slot"] == slot]
                if not slot_logs.empty:
                    latest = slot_logs.sort_values("timestamp").tail(1).iloc[0]
                    st.session_state.inputs[slot] = str(latest.get("입력항목", "") or "")
                    st.session_state.conditions[slot] = str(latest.get("컨디션", "") or "")
            st.session_state.last_items_df = today_logs.rename(columns={"slot": "슬롯", "date": "날짜"})
    except FileNotFoundError:
        pass

# ==================== CSV 유틸 ====================
def _parse_tags_from_slash(cell) -> List[str]:
    if pd.isna(cell):
        return []
    return [t.strip() for t in str(cell).split('/') if t.strip()]

def _parse_taglist_cell(cell: Any) -> List[str]:
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    s = "" if cell is None or (isinstance(cell, float) and pd.isna(cell)) else str(cell).strip()
    if not s or s == "[]":
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    s2 = s.strip().strip("[]")
    parts = [p.strip().strip("'").strip('"') for p in re.split(r"[,/]", s2) if p.strip()]
    return [p for p in parts if p]

def load_food_db_simple(path: str = FOOD_DB_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["식품", "등급", "태그(영양)"]:
        if c not in df.columns:
            df[c] = ""
    if "태그리스트" in df.columns:
        df["태그리스트"] = df["태그리스트"].apply(_parse_taglist_cell)
    else:
        df["태그리스트"] = df["태그(영양)"].apply(_parse_tags_from_slash)
    return df[["식품", "등급", "태그(영양)", "태그리스트"]]

def load_nutrient_dict_simple(path: str = NUTRIENT_DICT_CSV) -> Dict[str, str]:
    nd = pd.read_csv(path)
    for c in ["영양소", "한줄설명"]:
        if c not in nd.columns:
            nd[c] = ""
    return {str(r["영양소"]).strip(): str(r["한줄설명"]).strip() for _, r in nd.iterrows()}

# ==================== 분석 로직 ====================
def split_items(text: str) -> List[str]:
    if not text:
        return []
    first = [p.strip() for p in re.split(r"[,|\n|(|)]+", text) if p.strip()]
    final = []
    for part in first:
        final += [q.strip() for q in part.split('+') if q.strip()]
    return final

def parse_qty(token: str) -> Tuple[str, float]:
    m = re.search(r"(.*?)(\d+(?:\.\d+)?)\s*$", token)
    if m:
        return m.group(1).strip(), float(m.group(2))
    return token.strip(), 1.0

def match_item_to_foods(item: str, df_food: pd.DataFrame) -> pd.DataFrame:
    it = item.strip()
    hits = df_food[df_food["식품"].apply(lambda x: str(x).strip() in it or it in str(x).strip())].copy()
    hits = hits[hits["식품"].apply(lambda x: len(str(x).strip()) >= 1)]
    return hits

def analyze_items_for_slot(input_text: str, slot: str, df_food: pd.DataFrame,
                           nutrient_desc: Dict[str, str], condition: str = ""):
    raw_tokens = split_items(input_text)
    items = [parse_qty(tok) for tok in raw_tokens]

    per_item_rows, log_rows, unmatched_names = [], [], []
    nutrient_counts = defaultdict(float)

    for raw, qty in items:
        if not raw:
            continue
        matched = match_item_to_foods(raw, df_food)
        timestamp = datetime.now(TZ).isoformat(timespec="seconds")

        if matched.empty:
            per_item_rows.append({"슬롯": slot, "입력항목": raw, "수량": qty,
                                  "매칭식품": "", "등급": "", "태그": "", "컨디션": condition})
            log_rows.append({"timestamp": timestamp, "date": today_str(), "time": timestamp.split("T")[1],
                             "slot": slot, "입력항목": raw, "수량": qty, "매칭식품": "",
                             "등급": "", "태그": "", "컨디션": condition})
            unmatched_names.append(raw)
            continue

        agg_grade, tag_union, matched_names = "Safe", [], []
        for _, r in matched.iterrows():
            name = str(r["식품"]).strip()
            grade = str(r["등급"]).strip() or "Safe"
            tags = r.get("태그리스트", [])
            if not isinstance(tags, list):
                tags = _parse_taglist_cell(tags)
            agg_grade = "Avoid" if grade == "Avoid" else agg_grade
            matched_names.append(name)
            for t in tags:
                nutrient_counts[t] += qty
                tag_union.append(t)

        per_item_rows.append({"슬롯": slot, "입력항목": raw, "수량": qty,
                              "매칭식품": ", ".join(dict.fromkeys(matched_names)),
                              "등급": agg_grade, "태그": ", ".join(dict.fromkeys(tag_union)),
                              "컨디션": condition})
        log_rows.append({"timestamp": timestamp, "date": today_str(), "time": timestamp.split("T")[1],
                         "slot": slot, "입력항목": raw, "수량": qty,
                         "매칭식품": ", ".join(dict.fromkeys(matched_names)),
                         "등급": agg_grade, "태그": ", ".join(dict.fromkeys(tag_union)),
                         "컨디션": condition})
    return pd.DataFrame(per_item_rows), dict(nutrient_counts), pd.DataFrame(log_rows), unmatched_names

# ==================== 다음 식사 제안 ====================
def recommend_next_meal(nutrient_counts: Dict[str, float],
                        df_food: pd.DataFrame,
                        nutrient_desc: Dict[str, str],
                        threshold: float = 1.0,
                        top_nutrients: int = 3):
    # 전체 태그 리스트
    all_tags = sorted({t for lst in df_food["태그리스트"] for t in lst})
    deficits = {t: max(0, threshold - nutrient_counts.get(t, 0)) for t in all_tags if nutrient_counts.get(t, 0) < threshold}
    if not deficits:
        return [], []
    focus_tags = sorted(deficits.items(), key=lambda x: x[1], reverse=True)[:top_nutrients]

    suggestions, combos = [], []
    for tag, lack in focus_tags:
        desc = nutrient_desc.get(tag, "")
        foods = df_food[df_food["태그리스트"].apply(lambda lst: tag in lst)]["식품"].head(5).tolist()
        suggestions.append({"부족영양소": tag, "설명": desc, "추천식품": foods})
        combos += foods[:2]
    return suggestions, combos

# ==================== Streamlit UI ====================
def main():
    st.set_page_config(page_title="슬롯별 식단 분석 · 다음 식사 제안", page_icon="🥗", layout="centered")
    st.title("🥗 슬롯별 식단 분석 · 다음 식사 제안")

    init_daily_state()
    remain = next_midnight() - datetime.now(TZ)
    st.caption(f"현재 입력/결과는 자정까지 보존됩니다. 남은 시간: {remain.seconds//3600}시간 {remain.seconds%3600//60}분")

    df_food = load_food_db_simple(FOOD_DB_CSV)
    nutrient_desc = load_nutrient_dict_simple(NUTRIENT_DICT_CSV)
    d = st.date_input("기록 날짜", value=date.today())

    # 슬롯 입력
    for slot in SLOTS:
        val = st.text_area(slot, height=70, placeholder=f"{slot}에 먹은 것 입력",
                           key=f"ta_{slot}", value=st.session_state.inputs.get(slot, ""))
        st.session_state.inputs[slot] = val
        cond = st.text_input(f"{slot} 컨디션", placeholder="예: 양호 / 피곤함 / 복부팽만",
                             key=f"cond_{slot}", value=st.session_state.conditions.get(slot, ""))
        st.session_state.conditions[slot] = cond

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.number_input("충족 임계(수량합)", min_value=1, max_value=5,
                        value=st.session_state.get("threshold", 1),
                        step=1, key="threshold")
    with c2:
        st.checkbox("log.csv 저장", value=st.session_state.get("export_flag", True), key="export_flag")
    with c3:
        analyze_clicked = st.button("분석하기", type="primary")

    if analyze_clicked:
        all_items, all_logs, total_counts, all_unmatched = [], [], defaultdict(float), []
        for slot in SLOTS:
            items_df, counts, log_df, unmatched = analyze_items_for_slot(
                st.session_state.inputs.get(slot, ""), slot, df_food, nutrient_desc,
                condition=st.session_state.conditions.get(slot, "")
            )
            if not items_df.empty:
                items_df["날짜"] = d.isoformat()
            all_items.append(items_df)
            all_logs.append(log_df)
            all_unmatched += unmatched
            for k, v in counts.items():
                total_counts[k] += v

        items_df_all = pd.concat(all_items, ignore_index=True) if all_items else pd.DataFrame()
        logs_all = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()

        if st.session_state.export_flag and not logs_all.empty:
            try:
                try:
                    prev = pd.read_csv(LOG_CSV)
                    merged = pd.concat([prev, logs_all], ignore_index=True)
                except Exception:
                    merged = logs_all.copy()
                merged = merged.drop_duplicates(
                    subset=["date","slot","입력항목","매칭식품","등급","태그","컨디션"], keep="last"
                )
                merged.to_csv(LOG_CSV, index=False, encoding="utf-8-sig")
                st.success(f"log.csv 저장 완료 ✅")
            except Exception as e:
                st.error(f"log.csv 저장 오류: {e}")

        # ✅ 부족 영양소 기반 다음 식사 제안
        recs, combos = recommend_next_meal(total_counts, df_food, nutrient_desc, threshold=float(st.session_state.threshold))
        st.session_state.last_recs = recs
        st.session_state.last_combo = combos
        st.session_state.last_items_df = items_df_all

    # 출력
    st.markdown("### 🍱 슬롯별 매칭 결과")
    if st.session_state.last_items_df is None or st.session_state.last_items_df.empty:
        st.info("매칭된 항목이 없습니다.")
    else:
        cols = ["날짜","슬롯","입력항목","수량","매칭식품","등급","태그","컨디션"]
        st.dataframe(st.session_state.last_items_df[cols],
                     use_container_width=True,
                     height=min(420, 36*(len(st.session_state.last_items_df)+1)))

    st.markdown("### 🍽 다음 식사 제안")
    recs = st.session_state.get("last_recs", [])
    if recs is None or len(recs) == 0:
        st.warning("부족한 영양소가 없어 추천이 없습니다. threshold를 높여보세요.")
    else:
        for r in recs:
            st.write(f"- **{r['부족영양소']}** ({r['설명']}) → 추천식품: {', '.join(r['추천식품'])}")
        if st.session_state.last_combo:
            st.info("간단 조합 제안: " + " / ".join(st.session_state.last_combo[:4]))

if __name__ == "__main__":
    if st is None:
        print("This script requires Streamlit. Install with: pip install streamlit")
        sys.exit(1)
    main()