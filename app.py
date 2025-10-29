#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diet_analyzer_v4_condition_per_slot.py
→ 각 식사별 컨디션 입력 + 영양 불균형 + 컨디션 기반 다음 식사 제안
"""

from __future__ import annotations
import re, sys, ast, json
from collections import defaultdict
from datetime import datetime, date
from zoneinfo import ZoneInfo
import pandas as pd

try:
    import streamlit as st
except:
    st = None

# ================= 설정 =================
FOOD_DB_CSV = "food_db.csv"
NUTRIENT_DICT_CSV = "nutrient_dict.csv"
SLOTS = ["아침", "오전 간식", "점심", "오후 간식", "저녁"]
TZ = ZoneInfo("Europe/Paris")

# ================= 유틸 =================
def today_str(): return datetime.now(TZ).date().isoformat()

def _parse_taglist_cell(cell):
    if isinstance(cell, list): return cell
    if pd.isna(cell) or str(cell).strip() in ("", "[]"): return []
    s = str(cell).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list): return [str(x).strip() for x in v]
    except Exception: pass
    s2 = s.strip("[]")
    return [p.strip().strip("'").strip('"') for p in re.split(r"[,/]", s2) if p.strip()]

def load_food_db(): 
    df = pd.read_csv(FOOD_DB_CSV)
    if "태그리스트" not in df.columns:
        df["태그리스트"] = df["태그(영양)"].apply(_parse_taglist_cell)
    else:
        df["태그리스트"] = df["태그리스트"].apply(_parse_taglist_cell)
    return df

def load_nutrient_dict():
    nd = pd.read_csv(NUTRIENT_DICT_CSV)
    return {r["영양소"]: r["한줄설명"] for _, r in nd.iterrows()}

def split_items(text):
    if not text: return []
    return [p.strip() for p in re.split(r"[,|\n|(|)|+]+", text) if p.strip()]

def parse_qty(tok):
    m = re.search(r"(.*?)(\d+(?:\.\d+)?)$", tok)
    return (m.group(1).strip(), float(m.group(2))) if m else (tok.strip(), 1.0)

# ================= 분석 =================
def match_item_to_foods(item, df):
    i = item.strip()
    return df[df["식품"].apply(lambda x: i in str(x) or str(x) in i)]

def analyze_inputs(inputs, df_food):
    total = defaultdict(float)
    for slot, txt in inputs.items():
        for name, qty in [parse_qty(t) for t in split_items(txt)]:
            hits = match_item_to_foods(name, df_food)
            for _, r in hits.iterrows():
                for t in _parse_taglist_cell(r["태그리스트"]):
                    total[t] += qty
    return total

# ================= 컨디션 → 영양소 매핑 =================
COND_MAP = {
    "피곤": ["비타민B", "단백질"],
    "복부팽만": ["식이섬유", "저염"],
    "스트레스": ["마그네슘", "비타민C"],
    "수면": ["마그네슘", "칼륨"],
    "추움": ["철", "단백질"],
    "면역": ["비타민C", "비타민D"],
    "두통": ["마그네슘"],
    "소화": ["식이섬유", "저지방"]
}

def condition_weights(cond_inputs):
    weights = defaultdict(float)
    for slot, cond in cond_inputs.items():
        if not cond: continue
        for key, tags in COND_MAP.items():
            if key in cond:
                for t in tags:
                    weights[t] += 0.5
    return weights

# ================= 다음 식사 제안 =================
def recommend_next_meal(total_counts, cond_weights, df_food, nutrient_dict):
    if not total_counts: 
        return [{"부족영양소": "데이터 없음", "설명": "식사 기록이 없습니다.", "추천식품": []}]
    
    avg = sum(total_counts.values()) / (len(total_counts) or 1)
    ratios = {t: v / (avg or 1) for t, v in total_counts.items()}

    deficits = {t: 1.0 - r for t, r in ratios.items() if r < 0.8}
    for t, w in cond_weights.items():
        deficits[t] = deficits.get(t, 0) + w

    if not deficits:
        return [{"부족영양소": "균형 유지", "설명": "영양 균형이 양호합니다.", "추천식품": []}]

    sorted_tags = sorted(deficits.items(), key=lambda x: x[1], reverse=True)[:4]
    recs = []
    for tag, _ in sorted_tags:
        foods = df_food[df_food["태그리스트"].apply(lambda lst: tag in lst)]
        foods = foods[foods["등급"].isin(["Safe", "Caution"])]["식품"].head(5).tolist()
        desc = nutrient_dict.get(tag, "")
        if not foods: foods = ["(추천 식품 없음)"]
        recs.append({"부족영양소": tag, "설명": desc, "추천식품": foods})
    return recs

# ================= Streamlit UI =================
def main():
    st.set_page_config(page_title="AI 식단 분석기", page_icon="🥗")
    st.title("🥗 식단 + 컨디션 기반 다음 식사 제안")

    df_food = load_food_db()
    nutrient_dict = load_nutrient_dict()

    st.caption("쉼표로 구분하세요. 예: 닭가슴살, 현미밥, 김치")

    inputs, conditions = {}, {}
    for slot in SLOTS:
        with st.container():
            st.subheader(f"🍽 {slot}")
            inputs[slot] = st.text_area(f"{slot} 식단 입력", height=60, placeholder="예: 닭가슴살, 샐러드, 현미밥")
            conditions[slot] = st.text_input(f"{slot} 컨디션 (예: 피곤, 복부팽만, 스트레스 등)", key=f"cond_{slot}")

    if st.button("분석하기", type="primary"):
        total_counts = analyze_inputs(inputs, df_food)
        cond_weights = condition_weights(conditions)
        recs = recommend_next_meal(total_counts, cond_weights, df_food, nutrient_dict)

        df_summary = pd.DataFrame([{"영양소": k, "수량합": v} for k, v in total_counts.items()])
        if not df_summary.empty:
            avg = df_summary["수량합"].mean()
            df_summary["상태"] = df_summary["수량합"].apply(lambda x: "부족" if x < avg*0.8 else ("과다" if x > avg*1.2 else "적정"))
            st.markdown("### 🧭 영양소 요약")
            st.dataframe(df_summary, use_container_width=True)
        else:
            st.info("데이터 없음")

        st.markdown("### 🍽 다음 식사 제안")
        for r in recs:
            foods = ", ".join(r["추천식품"])
            st.write(f"- **{r['부족영양소']}** → {r['설명']}")
            st.caption(f"추천 식품: {foods}")

if __name__ == "__main__":
    if st is None:
        print("Streamlit 설치 필요: pip install streamlit")
        sys.exit(1)
    main()