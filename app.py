#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diet_analyzer.py (with condition-based personalized suggestions + clickable details)
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
    st = None  # allow import without Streamlit

# ====================== 설정 ======================
FOOD_DB_CSV = "food_db.csv"
NUTRIENT_DICT_CSV = "nutrient_dict.csv"
LOG_CSV = "log.csv"
FOOD_DB_UPDATED_CSV = "food_db_updated.csv"

SLOTS = ["아침", "아침보조제", "오전 간식", "점심", "점심보조제",
         "오후 간식", "저녁", "저녁보조제", "저녁 간식"]

TZ = ZoneInfo("Europe/Paris")

# ==================== 날짜/상태 관리 ====================
def today_str() -> str:
    return datetime.now(TZ).date().isoformat()

def next_midnight():
    now = datetime.now(TZ)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=TZ)

def init_daily_state():
    """자정 단위로 state를 유지. 날짜 바뀌면 자동 초기화."""
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

    # ✅ log.csv에서 오늘 날짜의 최신 1건씩 복원
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

# ==================== 유틸 ====================
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

_GRADE_ORDER = {"Avoid": 2, "Caution": 1, "Safe": 0}
def _worse_grade(g1: str, g2: str) -> str:
    return g1 if _GRADE_ORDER.get(g1, 0) >= _GRADE_ORDER.get(g2, 0) else g2
def _norm(s: str) -> str:
    return str(s or "").strip()

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

# ================== 분석 함수 ==================
def match_item_to_foods(item: str, df_food: pd.DataFrame) -> pd.DataFrame:
    it = _norm(item)
    hits = df_food[df_food["식품"].apply(lambda x: _norm(x) in it or it in _norm(x))].copy()
    hits = hits[hits["식품"].apply(lambda x: len(_norm(x)) >= 1)]
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
            unmatched_names.append(_norm(raw))
            continue

        agg_grade, tag_union, matched_names = "Safe", [], []
        for _, r in matched.iterrows():
            name = _norm(r["식품"])
            grade = _norm(r["등급"]) or "Safe"
            tags = r.get("태그리스트", [])
            if not isinstance(tags, list):
                tags = _parse_taglist_cell(tags)
            if not tags:
                tags = _parse_tags_from_slash(r.get("태그(영양)", ""))
            agg_grade = _worse_grade(agg_grade, grade)
            matched_names.append(name)
            for t in tags:
                if t:
                    tag_union.append(t)
                    nutrient_counts[t] += float(qty or 1.0)

        per_item_rows.append({"슬롯": slot, "입력항목": raw, "수량": qty,
                              "매칭식품": ", ".join(dict.fromkeys(matched_names)),
                              "등급": agg_grade, "태그": ", ".join(dict.fromkeys(tag_union)),
                              "컨디션": condition})
        log_rows.append({"timestamp": timestamp, "date": today_str(), "time": timestamp.split("T")[1],
                         "slot": slot, "입력항목": raw, "수량": qty,
                         "매칭식품": ", ".join(dict.fromkeys(matched_names)),
                         "등급": agg_grade, "태그": ", ".join(dict.fromkeys(tag_union)),
                         "컨디션": condition})
    return (pd.DataFrame(per_item_rows), dict(nutrient_counts),
            pd.DataFrame(log_rows), unmatched_names)

# ==================== URL 상태 저장/복원 ====================
def _b64_encode(obj: dict) -> str:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    comp = zlib.compress(raw, level=9)
    return base64.urlsafe_b64encode(comp).decode("ascii")

def _b64_decode(s: str) -> dict | None:
    try:
        comp = base64.urlsafe_b64decode(s.encode("ascii"))
        raw = zlib.decompress(comp)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None

def save_state_to_url():
    data = {
        "inputs": st.session_state.get("inputs", {}),
        "conditions": st.session_state.get("conditions", {}),
        "threshold": st.session_state.get("threshold", 1),
        "export_flag": st.session_state.get("export_flag", True),
        "daily_date": st.session_state.get("daily_date", None)
    }
    st.experimental_set_query_params(state=_b64_encode(data))

def load_state_from_url():
    params = st.experimental_get_query_params()
    if "state" not in params:
        return
    decoded = _b64_decode(params["state"][0])
    if not decoded:
        return
    st.session_state.inputs = decoded.get("inputs", {})
    st.session_state.conditions = decoded.get("conditions", {})
    st.session_state.threshold = decoded.get("threshold", 1)
    st.session_state.export_flag = decoded.get("export_flag", True)

# ================== 컨디션 → 태그 매핑 ==================
def condition_to_nutrients(condition: str) -> List[str]:
    cond = condition.lower()
    needs = []
    if any(k in cond for k in ["피곤", "무기력", "기운 없음"]):
        needs += ["단백질", "비타민B", "철분"]
    if any(k in cond for k in ["복부팽만", "더부룩", "소화불량"]):
        needs += ["저FODMAP", "식이섬유(적당량)"]
    if any(k in cond for k in ["속쓰림", "위산"]):
        needs += ["저지방", "저산성"]
    if "두통" in cond or "어지럽" in cond:
        needs += ["마그네슘", "수분"]
    if "불면" in cond or "수면" in cond:
        needs += ["트립토판", "칼슘"]
    if "변비" in cond:
        needs += ["식이섬유", "수분"]
    if "설사" in cond:
        needs += ["전해질", "수분"]
    return list(dict.fromkeys(needs))

# ================== 태그 → 식품군 추천 ==================
NUTRIENT_TO_FOODS = {
    "단백질": ["달걀", "닭가슴살", "두부", "그릭요거트", "생선"],
    "비타민B": ["현미", "통곡물빵", "콩류", "계란노른자"],
    "철분": ["시금치", "간", "붉은살생선", "렌틸콩"],
    "저FODMAP": ["호박", "당근", "감자", "쌀밥"],
    "식이섬유": ["귀리", "통곡물", "사과", "브로콜리"],
    "식이섬유(적당량)": ["당근", "호박죽", "바나나"],
    "저지방": ["찐감자", "닭가슴살", "두부", "저지방요거트"],
    "저산성": ["바나나", "감자", "두유", "흰죽"],
    "마그네슘": ["견과류", "시금치", "카카오닙스"],
    "수분": ["국물", "과일", "물", "수프"],
    "트립토판": ["달걀", "귀리", "바나나", "아보카도"],
    "칼슘": ["요거트", "멸치", "치즈", "두유"],
    "전해질": ["바나나", "소금간 국물", "미음"]
}

# ================== 식품 세부정보 표시 ==================
def show_food_details(food: str, df_food: pd.DataFrame, nutrient_desc: Dict[str, str]):
    matches = df_food[df_food["식품"].str.contains(food, case=False, na=False)]
    if matches.empty:
        st.warning(f"'{food}' 정보 없음")
        return
    with st.expander(f"🍽 {food} 세부정보 보기"):
        for _, row in matches.iterrows():
            grade = row.get("등급", "정보없음")
            tags = row.get("태그리스트", [])
            if not tags:
                tags = _parse_taglist_cell(row.get("태그(영양)", ""))
            st.write(f"**등급:** {grade}")
            st.write(f"**영양 태그:** {', '.join(tags) if tags else '없음'}")
            for t in tags:
                desc = nutrient_desc.get(t, "")
                if desc:
                    st.caption(f"• {t}: {desc}")

# ==================== Streamlit UI ====================
def main():
    st.set_page_config(page_title="슬롯별 식단 분석 · 다음 식사 제안", page_icon="🥗", layout="centered")
    st.title("🥗 슬롯별 식단 분석 · 다음 식사 제안")

    init_daily_state()
    remain = next_midnight() - datetime.now(TZ)
    st.caption(f"현재 입력/결과는 자정까지 보존됩니다. 남은 시간: {remain.seconds//3600}시간 {remain.seconds%3600//60}분")

    load_state_from_url()

    df_food = load_food_db_simple(FOOD_DB_CSV)
    nutrient_desc = load_nutrient_dict_simple(NUTRIENT_DICT_CSV)

    d = st.date_input("기록 날짜", value=date.today())

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
        analyze_clicked = st.button("분석하기", type="primary", key="analyze_btn")

    save_state_to_url()

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
                st.success("log.csv 저장 완료")
            except Exception as e:
                st.error(f"log.csv 저장 오류: {e}")

        st.session_state.last_items_df = items_df_all

        # ✅ 컨디션+영양 태그 기반 다음 식사 제안 (클릭형)
        st.markdown("### 🍽 개인화된 다음 식사 제안")

        total_tags = []
        if not items_df_all.empty and "태그" in items_df_all.columns:
            for tags in items_df_all["태그"].dropna():
                total_tags += [t.strip() for t in str(tags).split(",") if t.strip()]
        tag_counts = pd.Series(total_tags).value_counts().to_dict() if total_tags else {}

        suggestions = []
        for slot in SLOTS:
            cond = st.session_state.conditions.get(slot, "")
            if not cond.strip():
                continue
            needed_tags = condition_to_nutrients(cond)
            suggested_foods = []
            for tag in needed_tags:
                # 현재 태그 섭취량이 임계치보다 적으면 보완 제안
                if tag_counts.get(tag, 0) < st.session_state.threshold:
                    foods = NUTRIENT_TO_FOODS.get(tag, [])
                    if foods:
                        suggested_foods += foods
            suggestions.append((slot, cond, list(dict.fromkeys(suggested_foods[:5]))))

        if suggestions:
            for slot, cond, foods in suggestions:
                if foods:
                    st.markdown(f"#### 🩺 {slot} 컨디션: {cond}")
                    cols = st.columns(len(foods))
                    for i, food in enumerate(foods):
                        with cols[i]:
                            if st.button(food, key=f"suggest_btn_{slot}_{food}"):
                                show_food_details(food, df_food, nutrient_desc)
                else:
                    if cond.strip():
                        st.info(f"{slot} 컨디션({cond}) → 현재 식단 균형 양호")
        else:
            st.info("오늘 기록된 컨디션이 없습니다.")

    # ================== 결과 표 영역 ==================
    st.markdown("### 🍱 슬롯별 매칭 결과")
    if st.session_state.last_items_df is None or st.session_state.last_items_df.empty:
        st.info("매칭된 항목이 없습니다.")
    else:
        cols = ["날짜","슬롯","입력항목","수량","매칭식품","등급","태그"]
        if "컨디션" in st.session_state.last_items_df.columns:
            cols.append("컨디션")
        # 존재하는 컬럼만 표시(방어적)
        cols = [c for c in cols if c in st.session_state.last_items_df.columns]
        st.dataframe(
            st.session_state.last_items_df[cols],
            use_container_width=True,
            height=min(420, 36*(len(st.session_state.last_items_df)+1))
        )

if __name__ == "__main__":
    if st is None:
        print("This script requires Streamlit. Install with: pip install streamlit")
        sys.exit(1)
    main()