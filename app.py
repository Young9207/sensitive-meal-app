#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diet_analyzer.py (with next meal recommendation)
- 각 식사 슬롯별 컨디션 입력 가능
- log.csv / URL 상태 / 화면 표시 반영
- 부족 영양소 기반 다음 식사 제안 기능 추가
- 무한복사 방지 (최신 1건만 복원)
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

SLOTS = ["아침", "아침보조제", "오전 간식", "점심", "점심보조제",
         "오후 간식", "저녁", "저녁보조제", "저녁 간식"]

TZ = ZoneInfo("Europe/Paris")


# ==================== 날짜/상태 ====================
def today_str() -> str:
    return datetime.now(TZ).date().isoformat()

def next_midnight():
    now = datetime.now(TZ)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=TZ)

def init_daily_state():
    """자정 단위로 state 유지, 날짜 바뀌면 초기화"""
    if "daily_date" not in st.session_state:
        st.session_state.daily_date = today_str()
    if st.session_state.daily_date != today_str():
        for k in ["inputs", "conditions", "last_items_df", "last_nutri_df", "last_recs"]:
            st.session_state.pop(k, None)
        st.session_state.daily_date = today_str()

    st.session_state.setdefault("inputs", {s: "" for s in SLOTS})
    st.session_state.setdefault("conditions", {s: "" for s in SLOTS})
    st.session_state.setdefault("last_items_df", None)
    st.session_state.setdefault("last_nutri_df", None)
    st.session_state.setdefault("last_recs", [])
    st.session_state.setdefault("threshold", 1)
    st.session_state.setdefault("export_flag", True)

    # log.csv에서 오늘 날짜의 최신 입력만 복원
    try:
        df_log = pd.read_csv(LOG_CSV)
        today_logs = df_log[df_log["date"] == today_str()]
        for slot in SLOTS:
            slot_logs = today_logs[today_logs["slot"] == slot]
            if not slot_logs.empty:
                latest = slot_logs.sort_values("timestamp").tail(1).iloc[0]
                st.session_state.inputs[slot] = str(latest.get("입력항목", "") or "")
                st.session_state.conditions[slot] = str(latest.get("컨디션", "") or "")
    except FileNotFoundError:
        pass


# ==================== 유틸 ====================
def _parse_taglist_cell(cell: Any) -> List[str]:
    """CSV의 태그 문자열을 항상 리스트로 변환"""
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
    return [p.strip().strip("'").strip('"') for p in re.split(r"[,/]", s2) if p.strip()]

def load_food_db_simple(path: str = FOOD_DB_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "태그리스트" in df.columns:
        df["태그리스트"] = df["태그리스트"].apply(_parse_taglist_cell)
    else:
        df["태그리스트"] = df["태그(영양)"].apply(lambda x: _parse_taglist_cell(str(x)))
    return df[["식품", "등급", "태그리스트"]]

_GRADE_ORDER = {"Avoid": 2, "Caution": 1, "Safe": 0}
def _worse_grade(g1: str, g2: str) -> str:
    return g1 if _GRADE_ORDER.get(g1, 0) >= _GRADE_ORDER.get(g2, 0) else g2


# ================== 파서 및 매칭 ==================
def split_items(text: str) -> List[str]:
    """쉼표, 줄바꿈, + 로 분리"""
    if not text:
        return []
    parts = re.split(r"[,\n]+", text)
    items = []
    for p in parts:
        items += [q.strip() for q in p.split('+') if q.strip()]
    return items

def parse_qty(token: str) -> Tuple[str, float]:
    """토큰 끝 숫자 수량 파싱"""
    m = re.search(r"(.*?)(\d+(?:\.\d+)?)$", token.strip())
    if m:
        return m.group(1).strip(), float(m.group(2))
    return token.strip(), 1.0

def match_item_to_foods(item: str, df_food: pd.DataFrame) -> pd.DataFrame:
    it = item.strip()
    return df_food[df_food["식품"].apply(lambda x: it in x or x in it)]


# ================== 분석 및 추천 로직 ==================
def analyze_items_for_slot(text: str, slot: str, df_food: pd.DataFrame, condition: str = ""):
    tokens = split_items(text)
    items = [parse_qty(tok) for tok in tokens]
    rows, log_rows, unmatched = [], [], []
    nutrient_counts = defaultdict(float)

    for name, qty in items:
        if not name:
            continue
        matched = match_item_to_foods(name, df_food)
        timestamp = datetime.now(TZ).isoformat(timespec="seconds")
        if matched.empty:
            rows.append({"슬롯": slot, "입력항목": name, "수량": qty, "매칭식품": "",
                         "등급": "", "태그": "", "컨디션": condition})
            unmatched.append(name)
            continue

        agg_grade = "Safe"
        tags_all, matched_names = [], []
        for _, r in matched.iterrows():
            agg_grade = _worse_grade(agg_grade, r["등급"])
            matched_names.append(r["식품"])
            tags_all.extend(r["태그리스트"])
            for t in r["태그리스트"]:
                nutrient_counts[t] += qty

        rows.append({
            "슬롯": slot, "입력항목": name, "수량": qty,
            "매칭식품": ", ".join(matched_names),
            "등급": agg_grade, "태그": ", ".join(set(tags_all)),
            "컨디션": condition
        })
        log_rows.append({
            "timestamp": timestamp, "date": today_str(), "time": timestamp.split("T")[1],
            "slot": slot, "입력항목": name, "매칭식품": ", ".join(matched_names),
            "등급": agg_grade, "태그": ", ".join(set(tags_all)), "컨디션": condition
        })

    return pd.DataFrame(rows), dict(nutrient_counts), pd.DataFrame(log_rows), unmatched


def summarize_nutrients(counts: Dict[str, float], df_food: pd.DataFrame, threshold: int = 1):
    """태그별 수량합 및 부족판정"""
    all_tags = sorted({t for lst in df_food["태그리스트"] for t in lst})
    rows = []
    for tag in all_tags:
        val = counts.get(tag, 0)
        rows.append({"영양소": tag, "수량합": val, "상태": "충족" if val >= threshold else "부족"})
    return pd.DataFrame(rows).sort_values(["상태", "수량합"], ascending=[True, False])


def _tag_deficits(counts: Dict[str, float], tags: List[str], threshold: int = 1):
    """부족한 태그만 추출"""
    return {t: max(0, threshold - counts.get(t, 0)) for t in tags if counts.get(t, 0) < threshold}

def _food_score(tags, deficits, grade):
    """태그 일치 + 등급 가중치로 점수 산정"""
    gain = sum(deficits.get(t, 0) for t in tags)
    grade_w = {"Safe": 1.0, "Caution": 0.6, "Avoid": 0.0}[grade]
    return gain * grade_w

def recommend_next_meal(counts, df_food, threshold=1, top_nutrients=3, per_food=5):
    all_tags = sorted({t for lst in df_food["태그리스트"] for t in lst})
    deficits = _tag_deficits(counts, all_tags, threshold)
    if not deficits:
        return []
    focus_tags = sorted(deficits.items(), key=lambda x: x[1], reverse=True)[:top_nutrients]
    focus_tags = [t for t, _ in focus_tags]
    recs = []
    for tag in focus_tags:
        cand = df_food[df_food["태그리스트"].apply(lambda lst: tag in lst)].copy()
        cand["score"] = cand.apply(lambda r: _food_score(r["태그리스트"], deficits, r["등급"]), axis=1)
        top_foods = cand.sort_values("score", ascending=False).head(per_food)["식품"].tolist()
        recs.append({"부족영양소": tag, "추천식품": top_foods})
    return recs


# ==================== UI ====================
def main():
    st.set_page_config(page_title="식단 분석 + 다음 식사 제안", layout="centered")
    st.title("🥗 식단 분석 · 다음 식사 제안")

    init_daily_state()
    remain = next_midnight() - datetime.now(TZ)
    st.caption(f"현재 상태는 자정까지 유지됩니다. 남은 시간: {remain.seconds//3600}시간 {remain.seconds%3600//60}분")

    df_food = load_food_db_simple()
    d = st.date_input("기록 날짜", value=date.today())

    for slot in SLOTS:
        val = st.text_area(slot, height=70, placeholder=f"{slot}에 먹은 것 입력",
                           key=f"ta_{slot}", value=st.session_state.inputs.get(slot, ""))
        st.session_state.inputs[slot] = val
        cond = st.text_input(f"{slot} 컨디션", placeholder="예: 양호 / 피곤함 / 복부팽만",
                             key=f"cond_{slot}", value=st.session_state.conditions.get(slot, ""))
        st.session_state.conditions[slot] = cond

    analyze_clicked = st.button("분석하기", type="primary")

    if analyze_clicked:
        total_counts, all_items = defaultdict(float), []
        for slot in SLOTS:
            items_df, counts, _, _ = analyze_items_for_slot(
                st.session_state.inputs[slot], slot, df_food, st.session_state.conditions[slot]
            )
            for k, v in counts.items():
                total_counts[k] += v
            all_items.append(items_df)
        items_df_all = pd.concat(all_items, ignore_index=True)
        st.session_state.last_items_df = items_df_all
        st.session_state.last_nutri_df = summarize_nutrients(total_counts, df_food, threshold=st.session_state.threshold)
        st.session_state.last_recs = recommend_next_meal(total_counts, df_food, threshold=st.session_state.threshold)

    st.markdown("### 🍱 식단 결과")
    if st.session_state.last_items_df is not None:
        st.dataframe(st.session_state.last_items_df, use_container_width=True)

    st.markdown("### 🧭 영양 요약")
    if st.session_state.last_nutri_df is not None:
        st.dataframe(st.session_state.last_nutri_df, use_container_width=True)

    st.markdown("### 🍽 다음 식사 제안")
    recs = st.session_state.last_recs
    if not recs:
        st.success("부족한 영양소가 없습니다. 균형 잡힌 식사예요!")
    else:
        for r in recs:
            st.write(f"- **{r['부족영양소']}** 보완을 위해 추천: {', '.join(r['추천식품'])}")

if __name__ == "__main__":
    if st is None:
        print("This script requires Streamlit. Run with: streamlit run diet_analyzer.py")
        sys.exit(1)
    main()