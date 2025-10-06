#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diet_analyzer.py
간단 입력 → 식품 매칭 → 영양 분석 → 다음 식사 제안
- 필요 파일: food_db.csv (식품, 등급, 태그(영양)), nutrient_dict.csv(영양소, 한줄설명 ...)
- 실행: streamlit run diet_analyzer.py
"""

import re
import sys
from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None  # allow import without Streamlit


# ====================== 설정 ======================
FOOD_DB_CSV = "food_db.csv"
NUTRIENT_DICT_CSV = "nutrient_dict.csv"


# ==================== 유틸/전처리 ====================
def _parse_tags(cell) -> List[str]:
    if pd.isna(cell):
        return []
    return [t.strip() for t in str(cell).split('/') if t.strip()]


def load_food_db_simple(path: str = FOOD_DB_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "태그리스트" not in df.columns:
        df["태그리스트"] = df["태그(영양)"].apply(_parse_tags)
    for c in ["식품", "등급"]:
        if c not in df.columns:
            df[c] = ""
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


def split_items(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[,|\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def _norm(s: str) -> str:
    return str(s or "").strip()


def match_item_to_foods(item: str, df_food: pd.DataFrame) -> pd.DataFrame:
    """item(예: '소고기 미역국')에 대해 food_db의 식품명이 포함되면 매칭.
       반대방향(항목명이 더 짧고 DB가 길 때)도 허용."""
    it = _norm(item)
    hits = df_food[
        df_food["식품"].apply(lambda x: _norm(x) in it or it in _norm(x))
    ].copy()
    hits = hits[hits["식품"].apply(lambda x: len(_norm(x)) >= 1)]
    return hits


def analyze_diet(input_text: str, df_food: pd.DataFrame, nutrient_desc: Dict[str, str], threshold: int = 1):
    """입력 텍스트 → (항목별 매칭표, 영양소 요약표, 태그별 카운트 딕셔너리)"""
    items = split_items(input_text)
    per_item_rows = []
    nutrient_counts = defaultdict(float)

    for raw in items:
        matched = match_item_to_foods(raw, df_food)
        if matched.empty:
            per_item_rows.append({
                "입력항목": raw, "매칭식품": "", "등급": "", "태그": ""
            })
            continue

        agg_grade = "Safe"
        tag_union = []
        matched_names = []

        for _, r in matched.iterrows():
            name = _norm(r["식품"])
            grade = _norm(r["등급"]) or "Safe"
            tags = list(r.get("태그리스트", [])) or _parse_tags(r.get("태그(영양)", ""))

            agg_grade = _worse_grade(agg_grade, grade)
            matched_names.append(name)
            for t in tags:
                if t:
                    tag_union.append(t)
                    nutrient_counts[t] += 1.0

        per_item_rows.append({
            "입력항목": raw,
            "매칭식품": ", ".join(dict.fromkeys(matched_names)),
            "등급": agg_grade,
            "태그": ", ".join(dict.fromkeys(tag_union))
        })

    # 영양소 요약 테이블 구성
    all_tags = sorted({t for tlist in df_food["태그리스트"] for t in tlist})
    rows = []
    for tag in all_tags:
        cnt = float(nutrient_counts.get(tag, 0))
        rows.append({
            "영양소": tag,
            "횟수": cnt,
            "상태": "충족" if cnt >= threshold else "부족",
            "한줄설명": nutrient_desc.get(tag, "")
        })
    nutrient_df = pd.DataFrame(rows).sort_values(["상태", "횟수", "영양소"], ascending=[True, False, True])
    items_df = pd.DataFrame(per_item_rows)
    return items_df, nutrient_df, nutrient_counts


def recommend_next_meal(nutrient_counts: Dict[str, float], df_food: pd.DataFrame, nutrient_desc: Dict[str, str],
                        top_nutrients: int = 2, per_food: int = 4):
    """부족 영양소 중심 추천: Safe 식품 우선 + 간단 조합"""
    tag_universe = {tt for lst in df_food["태그리스트"] for tt in lst}
    tag_totals = {t: float(nutrient_counts.get(t, 0)) for t in tag_universe}
    lacking = [t for t, v in sorted(tag_totals.items(), key=lambda x: x[1]) if v < 1.0]
    lacking = lacking[:top_nutrients]

    suggestions = []
    for tag in lacking:
        pool = df_food[(df_food["등급"] == "Safe") & (df_food["태그리스트"].apply(lambda lst: tag in lst))]
        foods = pool["식품"].dropna().astype(str).head(per_food).tolist()
        suggestions.append({
            "부족영양소": tag,
            "설명": nutrient_desc.get(tag, ""),
            "추천식품": foods
        })

    combo = []
    for s in suggestions:
        for f in s["추천식품"]:
            if f not in combo:
                combo.append(f)
            if len(combo) >= 4:
                break
        if len(combo) >= 4:
            break

    return suggestions, combo


# ==================== Streamlit UI ====================
def main():
    try:
        import streamlit as st
    except Exception as e:
        print("This script requires Streamlit to run the UI. Install with: pip install streamlit")
        sys.exit(1)

    st.set_page_config(page_title="간단 식단 분석 · 다음 식사 제안", page_icon="🥗", layout="centered")
    st.title("🥗 간단 식단 분석 · 다음 식사 제안")

    # 파일 로딩
    with st.expander("데이터 파일 경로 설정", expanded=False):
        food_path = st.text_input("food_db.csv 경로", value=FOOD_DB_CSV)
        nutri_path = st.text_input("nutrient_dict.csv 경로", value=NUTRIENT_DICT_CSV)
        load_btn = st.button("파일 다시 로드")

    if load_btn:
        st.experimental_rerun()

    # 실제 로딩
    try:
        df_food = load_food_db_simple(food_path if 'food_path' in locals() else FOOD_DB_CSV)
        nutrient_desc = load_nutrient_dict_simple(nutri_path if 'nutri_path' in locals() else NUTRIENT_DICT_CSV)
    except Exception as e:
        st.error(f"CSV 로딩 중 오류: {e}")
        st.stop()

    st.markdown("— 입력 예시: `소고기 미역국, 현미밥, 연어구이`")
    user_input = st.text_area("식단 텍스트 (쉼표/줄바꿈 구분)", height=100, placeholder="예: 소고기 미역국, 현미밥, 연어구이")

    col1, col2 = st.columns([1,1])
    with col1:
        threshold = st.number_input("충족 임계(횟수)", min_value=1, max_value=5, value=1, step=1, help="영양소를 '충족'으로 표시할 최소 횟수")
    with col2:
        if st.button("분석하기", type="primary"):
            try:
                items_df, nutrient_df, counts = analyze_diet(user_input, df_food, nutrient_desc, threshold=int(threshold))
                st.markdown("### 🍱 항목별 매칭 결과")
                if items_df.empty:
                    st.info("매칭된 항목이 없습니다.")
                else:
                    st.dataframe(items_df, use_container_width=True, height=min(320, 36 * (len(items_df) + 1)))

                st.markdown("### 🧭 영양 태그 요약 (충족/부족 + 한줄설명)")
                if nutrient_df.empty:
                    st.info("영양소 사전 또는 태그 정보가 비어 있습니다.")
                else:
                    st.dataframe(nutrient_df, use_container_width=True, height=min(420, 36 * (len(nutrient_df) + 1)))

                st.markdown("### 🍽 다음 식사 제안 (부족 보완용)")
                recs, combo = recommend_next_meal(counts, df_food, nutrient_desc, top_nutrients=2, per_food=4)
                if not recs:
                    st.success("핵심 부족 영양소가 없습니다. 균형이 잘 맞았어요!")
                else:
                    for r in recs:
                        foods_text = ", ".join(r["추천식품"]) if r["추천식품"] else "(추천 식품 없음)"
                        st.write(f"- **{r['부족영양소']}**: {r['설명']}")
                        st.caption(f"  추천 식품: {foods_text}")
                    if combo:
                        st.info("간단 조합 제안: " + " / ".join(combo[:4]))
            except Exception as e:
                st.error(f"분석 중 오류: {e}")


if __name__ == "__main__":
    # Allow importing functions without running Streamlit
    if st is None:
        # Not running Streamlit – expose functions
        pass
    else:
        main()
