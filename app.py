#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diet_analyzer_v5.py
-------------------
- nutrient_dict.csv (5컬럼 구조) 완전 지원
- 항상 다음 식사 제안 표시
- food_db.csv 매칭 기반 영양 분석
"""

from __future__ import annotations
import re, sys, ast, json
from collections import defaultdict
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
SLOTS = ["아침", "오전 간식", "점심", "오후 간식", "저녁"]
TZ = ZoneInfo("Europe/Paris")

# ==================== 날짜 관리 ====================
def today_str() -> str:
    return datetime.now(TZ).date().isoformat()

def next_midnight():
    now = datetime.now(TZ)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=TZ)

def init_daily_state():
    """자정 기준 상태 초기화"""
    if "daily_date" not in st.session_state:
        st.session_state.daily_date = today_str()
    if st.session_state.daily_date != today_str():
        for k in ["inputs", "conditions", "last_items_df", "last_recs"]:
            st.session_state.pop(k, None)
        st.session_state.daily_date = today_str()
    st.session_state.setdefault("inputs", {s: "" for s in SLOTS})
    st.session_state.setdefault("conditions", {s: "" for s in SLOTS})
    st.session_state.setdefault("last_items_df", None)
    st.session_state.setdefault("last_recs", [])
    st.session_state.setdefault("threshold", 1)
    st.session_state.setdefault("export_flag", True)

# ==================== 유틸 ====================
def _parse_taglist_cell(cell):
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    s = str(cell).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v]
    except Exception:
        pass
    s2 = s.strip().strip("[]")
    parts = [p.strip().strip("'").strip('"') for p in re.split(r"[,/]", s2) if p.strip()]
    return [p for p in parts if p]

def load_food_db_simple(path=FOOD_DB_CSV):
    df = pd.read_csv(path)
    df["태그리스트"] = df["태그리스트"].apply(_parse_taglist_cell) if "태그리스트" in df.columns else df["태그(영양)"].apply(_parse_taglist_cell)
    return df

def load_nutrient_dict(path=NUTRIENT_DICT_CSV):
    df = pd.read_csv(path)
    if "영양소" not in df.columns:
        raise ValueError("nutrient_dict.csv에 '영양소' 컬럼이 필요합니다.")
    out = {}
    for _, r in df.iterrows():
        out[str(r["영양소"]).strip()] = {
            "한줄설명": str(r.get("한줄설명", "")),
            "자세한설명": str(r.get("자세한설명", "")),
            "혜택라벨": str(r.get("혜택라벨(요약)", "")),
            "대표식품": str(r.get("대표식품(쉼표로구분)", ""))
        }
    return out

def split_items(text):
    if not text:
        return []
    first = [p.strip() for p in re.split(r"[,|\n]+", text) if p.strip()]
    final = []
    for p in first:
        final += [q.strip() for q in p.split('+') if q.strip()]
    return final

def parse_qty(token):
    m = re.search(r"(.*?)(\d+(?:\.\d+)?)\s*$", token)
    if m:
        return m.group(1).strip(), float(m.group(2))
    return token.strip(), 1.0

# ==================== 분석 ====================
def match_item_to_foods(item, df_food):
    it = str(item).strip()
    hits = df_food[df_food["식품"].apply(lambda x: it in str(x) or str(x) in it)]
    return hits

def analyze_items_for_slot(text, slot, df_food, condition=""):
    tokens = split_items(text)
    nutrient_counts = defaultdict(float)
    rows, unmatched = [], []
    for raw in tokens:
        name, qty = parse_qty(raw)
        matched = match_item_to_foods(name, df_food)
        if matched.empty:
            unmatched.append(name)
            rows.append({"슬롯": slot, "입력항목": name, "수량": qty, "매칭식품": "", "등급": "", "태그": "", "컨디션": condition})
            continue
        tags = []
        for _, r in matched.iterrows():
            tlist = _parse_taglist_cell(r.get("태그리스트", ""))
            for t in tlist:
                nutrient_counts[t] += qty
            tags += tlist
        rows.append({
            "슬롯": slot, "입력항목": name, "수량": qty,
            "매칭식품": ", ".join(matched["식품"].tolist()),
            "등급": ", ".join(matched["등급"].tolist()),
            "태그": ", ".join(tags), "컨디션": condition
        })
    return pd.DataFrame(rows), dict(nutrient_counts), unmatched

# ==================== 다음 식사 제안 ====================
def recommend_next_meal(nutrient_counts, df_food, nutrient_dict, threshold=1, top_nutrients=3):
    all_tags = sorted({t for lst in df_food["태그리스트"] for t in lst})
    if not all_tags:
        return [{"영양소": "데이터 없음", "설명": "food_db.csv에 태그 정보가 없습니다.", "추천식품": []}]
    deficits = {t: max(0, threshold - nutrient_counts.get(t, 0)) for t in all_tags}
    # 부족한 태그 우선 정렬
    lack_tags = sorted(deficits.items(), key=lambda x: x[1], reverse=True)[:top_nutrients]
    # 부족 없으면 기본 Safe 추천
    if not any(v > 0 for v in deficits.values()):
        sample_foods = df_food[df_food["등급"] == "Safe"]["식품"].head(5).tolist()
        return [{
            "영양소": "균형 유지",
            "설명": "모든 영양소 충족",
            "대표식품": ", ".join(sample_foods),
            "혜택": "유지",
            "추천식품": sample_foods
        }]
    recs = []
    for tag, lack in lack_tags:
        info = nutrient_dict.get(tag, {})
        foods = df_food[df_food["태그리스트"].apply(lambda lst: tag in lst)]["식품"].head(5).tolist()
        recs.append({
            "영양소": tag,
            "설명": info.get("한줄설명", ""),
            "혜택": info.get("혜택라벨", ""),
            "대표식품": info.get("대표식품", ""),
            "추천식품": foods
        })
    return recs

# ==================== Streamlit ====================
def main():
    st.set_page_config(page_title="식단 분석 + 다음 식사 제안 (v5)", page_icon="🥗")
    st.title("🥗 식단 분석 + 다음 식사 제안 (v5)")

    init_daily_state()
    remain = next_midnight() - datetime.now(TZ)
    st.caption(f"현재 입력은 자정까지 유지됩니다. 남은 시간: {remain.seconds//3600}시간 {remain.seconds%3600//60}분")

    df_food = load_food_db_simple(FOOD_DB_CSV)
    nutrient_dict = load_nutrient_dict(NUTRIENT_DICT_CSV)
    d = st.date_input("기록 날짜", value=date.today())

    for slot in SLOTS:
        val = st.text_area(slot, height=70, key=f"ta_{slot}", value=st.session_state.inputs.get(slot, ""))
        st.session_state.inputs[slot] = val
        cond = st.text_input(f"{slot} 컨디션", key=f"cond_{slot}", value=st.session_state.conditions.get(slot, ""))
        st.session_state.conditions[slot] = cond

    st.number_input("충족 임계(수량합)", 1, 5, st.session_state.get("threshold", 1), key="threshold")
    analyze_clicked = st.button("분석하기", type="primary")

    if analyze_clicked:
        all_items, total_counts = [], defaultdict(float)
        for slot in SLOTS:
            items_df, counts, _ = analyze_items_for_slot(
                st.session_state.inputs[slot], slot, df_food,
                condition=st.session_state.conditions.get(slot, "")
            )
            if not items_df.empty:
                items_df["날짜"] = d.isoformat()
                all_items.append(items_df)
            for k, v in counts.items():
                total_counts[k] += v
        items_all = pd.concat(all_items, ignore_index=True) if all_items else pd.DataFrame()
        st.session_state.last_items_df = items_all
        st.session_state.last_recs = recommend_next_meal(total_counts, df_food, nutrient_dict, threshold=st.session_state.threshold)

    # 출력
    st.markdown("### 🍱 슬롯별 매칭 결과")
    if st.session_state.last_items_df is None or st.session_state.last_items_df.empty:
        st.info("매칭된 항목이 없습니다.")
    else:
        st.dataframe(st.session_state.last_items_df, use_container_width=True)

    st.markdown("### 🍽 다음 식사 제안")
    recs = st.session_state.get("last_recs", [])
    if not recs:
        st.warning("추천할 식품이 없습니다.")
    else:
        for r in recs:
            foods = ", ".join(r.get("추천식품", []))
            st.write(f"- **{r['영양소']}** ({r['혜택']}) → {r['설명']}")
            if foods:
                st.caption(f"  🔹 추천식품: {foods}")
            if r.get("대표식품"):
                st.caption(f"  🥗 대표식품 예시: {r['대표식품']}")

if __name__ == "__main__":
    if st is None:
        print("This script requires Streamlit. Install with: pip install streamlit")
        sys.exit(1)
    main()