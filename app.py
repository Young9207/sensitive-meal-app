#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diet_analyzer_v6.py

📘 주요 기능
- 아침~저녁 슬롯별 식단 입력 + 컨디션 입력
- 식품 매칭 (food_db.csv 기반)
- 영양소 분석 및 부족 태그 요약
- 다음 식사 제안 (Safe, Caution 중심)
- log.csv 자동 저장 및 food_db 업데이트

필요 파일:
- food_db.csv : 식품, 등급, 태그(영양), 태그리스트
- nutrient_dict.csv : 영양소, 한줄설명
실행:
    streamlit run diet_analyzer_v6.py
"""

from __future__ import annotations
import re, ast, json, base64, zlib, sys
from datetime import datetime, date, timedelta
from collections import defaultdict
from typing import List, Dict, Tuple, Any
from zoneinfo import ZoneInfo
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None


# ====================== 기본 설정 ======================
FOOD_DB_CSV = "food_db.csv"
NUTRIENT_DICT_CSV = "nutrient_dict.csv"
LOG_CSV = "log.csv"
FOOD_DB_UPDATED_CSV = "food_db_updated.csv"
TZ = ZoneInfo("Europe/Paris")

SLOTS = ["아침", "오전 간식", "점심", "오후 간식", "저녁"]
CONDITIONS = ["피로감", "복부팽만", "집중도", "기분", "소화상태"]


# ====================== 유틸 ======================
def today_str() -> str:
    return datetime.now(TZ).date().isoformat()

def _parse_taglist_cell(cell: Any) -> List[str]:
    """CSV의 태그리스트 셀을 항상 리스트로 변환."""
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    s = str(cell or "").strip()
    if not s or s == "[]":
        return []
    for parser in (ast.literal_eval, json.loads):
        try:
            v = parser(s)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
    parts = [p.strip().strip("'").strip('"') for p in re.split(r"[,/]", s) if p.strip()]
    return parts

def load_food_db(path: str = FOOD_DB_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "태그리스트" not in df.columns:
        df["태그리스트"] = df["태그(영양)"].apply(lambda x: _parse_taglist_cell(x))
    else:
        df["태그리스트"] = df["태그리스트"].apply(_parse_taglist_cell)
    for c in ["식품", "등급", "태그(영양)"]:
        if c not in df.columns:
            df[c] = ""
    return df[["식품", "등급", "태그(영양)", "태그리스트"]]

def load_nutrient_dict(path: str = NUTRIENT_DICT_CSV) -> Dict[str, str]:
    df = pd.read_csv(path)
    return {str(r["영양소"]).strip(): str(r["한줄설명"]).strip() for _, r in df.iterrows()}


# ====================== 파서 ======================
def split_items(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[,|\n|(|)]+", text) if p.strip()]
    tokens = []
    for p in parts:
        tokens += [q.strip() for q in p.split('+') if q.strip()]
    return tokens

def parse_qty(token: str) -> Tuple[str, float]:
    m = re.search(r"(.*?)(\d+(?:\.\d+)?)\s*$", token)
    if m:
        return m.group(1).strip(), float(m.group(2))
    return token.strip(), 1.0


# ====================== 매칭 및 분석 ======================
def match_item_to_foods(item: str, df_food: pd.DataFrame) -> pd.DataFrame:
    it = str(item).strip()
    hits = df_food[df_food["식품"].apply(lambda x: str(x).strip() in it or it in str(x).strip())]
    return hits

def analyze_items_for_slot(input_text: str, slot: str, df_food: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    tokens = split_items(input_text)
    pairs = [parse_qty(t) for t in tokens]
    rows = []
    nutrient_counts = defaultdict(float)
    for name, qty in pairs:
        if not name:
            continue
        matched = match_item_to_foods(name, df_food)
        if matched.empty:
            rows.append({"슬롯": slot, "입력항목": name, "수량": qty, "매칭식품": "", "등급": "", "태그": ""})
            continue
        tags_union = []
        grade = "Safe"
        for _, r in matched.iterrows():
            grade = r["등급"] or "Safe"
            tags = _parse_taglist_cell(r["태그리스트"])
            for t in tags:
                nutrient_counts[t] += qty
                tags_union.append(t)
        rows.append({
            "슬롯": slot,
            "입력항목": name,
            "수량": qty,
            "매칭식품": ", ".join(matched["식품"]),
            "등급": grade,
            "태그": ", ".join(set(tags_union))
        })
    return pd.DataFrame(rows), nutrient_counts

def summarize_nutrients(total_counts: Dict[str, float], df_food: pd.DataFrame, nutrient_dict: Dict[str, str], threshold=1) -> pd.DataFrame:
    tags = sorted({t for lst in df_food["태그리스트"] for t in lst})
    result = []
    for tag in tags:
        val = total_counts.get(tag, 0.0)
        result.append({
            "영양소": tag,
            "수량합": val,
            "상태": "충족" if val >= threshold else "부족",
            "한줄설명": nutrient_dict.get(tag, "")
        })
    return pd.DataFrame(result)


# ====================== 다음 식사 제안 ======================
def recommend_next_meal(total_counts: Dict[str, float], df_food: pd.DataFrame,
                        nutrient_dict: Dict[str, str], threshold=1) -> Tuple[list, list]:
    tags = sorted({t for lst in df_food["태그리스트"] for t in lst})
    deficits = {t: threshold - total_counts.get(t, 0.0) for t in tags if total_counts.get(t, 0.0) < threshold}
    if not deficits:
        safe_foods = df_food[df_food["등급"] == "Safe"]["식품"].head(3).tolist()
        return ([{"부족영양소": "균형 유지", "설명": "모든 영양소가 충족되었습니다.", "추천식품": safe_foods}], safe_foods)
    recs, combo = [], []
    for tag in sorted(deficits, key=lambda x: deficits[x], reverse=True)[:3]:
        foods = df_food[df_food["태그리스트"].apply(lambda lst: tag in lst)]
        foods = foods[foods["등급"].isin(["Safe", "Caution"])].head(5)["식품"].tolist()
        recs.append({"부족영양소": tag, "설명": nutrient_dict.get(tag, ""), "추천식품": foods})
        combo += foods[:2]
    return recs, combo[:4]


# ====================== Streamlit UI ======================
def main():
    st.set_page_config(page_title="식단 분석 & 다음 식사 제안", page_icon="🥗", layout="centered")
    st.title("🥗 식단 분석 & 다음 식사 제안")

    df_food = load_food_db()
    nutrient_dict = load_nutrient_dict()

    st.caption("예: 닭가슴살, 현미밥, 김치, 아메리카노")

    inputs, conditions = {}, {}
    for slot in SLOTS:
        with st.expander(f"🍽 {slot}", expanded=False):
            inputs[slot] = st.text_area(f"{slot} 식단 입력", key=f"food_{slot}", height=60)
            st.markdown("#### 컨디션 체크")
            cond = {}
            for c in CONDITIONS:
                cond[c] = st.slider(f"{c}", 0, 10, 5, key=f"{slot}_{c}")
            conditions[slot] = cond

    threshold = st.number_input("영양 충족 임계값", 1, 5, 1)
    if st.button("분석하기", type="primary"):
        all_rows, total_counts = [], defaultdict(float)
        for slot in SLOTS:
            df_slot, counts = analyze_items_for_slot(inputs[slot], slot, df_food)
            all_rows.append(df_slot)
            for k, v in counts.items():
                total_counts[k] += v
        df_all = pd.concat(all_rows, ignore_index=True)
        st.subheader("📋 매칭 결과")
        st.dataframe(df_all)

        st.subheader("🧭 영양 요약")
        nutri_df = summarize_nutrients(total_counts, df_food, nutrient_dict, threshold)
        st.dataframe(nutri_df)

        st.subheader("🍽 다음 식사 제안")
        recs, combo = recommend_next_meal(total_counts, df_food, nutrient_dict, threshold)
        for r in recs:
            st.markdown(f"**{r['부족영양소']}**: {r['설명']}")
            st.caption("추천 식품: " + ", ".join(r["추천식품"]))
        st.info("간단 조합 제안: " + " / ".join(combo))


if __name__ == "__main__":
    if st is None:
        print("⚠️ Streamlit이 필요합니다. 설치: pip install streamlit")
        sys.exit(1)
    else:
        main()