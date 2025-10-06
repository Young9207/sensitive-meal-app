# ==== 간단 분석 + 다음 식사 제안 (food_db.csv, nutrient_dict.csv 기반) ====
# 필요한 입력: food_db.csv (식품/등급/태그(영양)); nutrient_dict.csv(영양소/한줄설명 ...)
# 사용 예: Streamlit 탭/섹션 하나에 이 블록을 붙여 넣으면 동작합니다.

import pandas as pd
import re
from collections import defaultdict

# ---- Streamlit 섹션 (앱 어딘가에 추가) ----
import streamlit as st

# --- 설정: 파일 경로 (앱의 기존 경로를 쓰고 싶다면 아래 두 변수를 바꿔 주세요)
FOOD_DB_CSV = "/mnt/data/food_db.csv"          # 예시 경로
NUTRIENT_DICT_CSV = "/mnt/data/nutrient_dict.csv"

# --- 로딩 & 전처리
def _parse_tags(cell):
    if pd.isna(cell):
        return []
    return [t.strip() for t in str(cell).split('/') if t.strip()]

def load_food_db_simple(path=FOOD_DB_CSV):
    df = pd.read_csv(path)
    if "태그리스트" not in df.columns:
        df["태그리스트"] = df["태그(영양)"].apply(_parse_tags)
    # 최소 컬럼 보정
    for c in ["식품","등급"]:
        if c not in df.columns:
            df[c] = ""
    return df[["식품","등급","태그(영양)","태그리스트"]]

def load_nutrient_dict_simple(path=NUTRIENT_DICT_CSV):
    nd = pd.read_csv(path)
    # 필수 컬럼 보정
    for c in ["영양소","한줄설명"]:
        if c not in nd.columns:
            nd[c] = ""
    # 매핑 딕셔너리: 영양소 -> 한줄설명
    return {str(r["영양소"]).strip(): str(r["한줄설명"]).strip() for _, r in nd.iterrows()}

FOOD_DB_SIMPLE = load_food_db_simple()
NUTRIENT_DESC = load_nutrient_dict_simple()

_GRADE_ORDER = {"Avoid": 2, "Caution": 1, "Safe": 0}
def _worse_grade(g1, g2):
    return g1 if _GRADE_ORDER.get(g1, 0) >= _GRADE_ORDER.get(g2, 0) else g2

# --- 문자열 분리/정규화
def split_items(text: str):
    if not text:
        return []
    parts = re.split(r"[,|\n]+", text)
    return [p.strip() for p in parts if p.strip()]

def _norm(s: str):
    return str(s or "").strip()

# --- 부분 일치 매칭: 항목 내부 재료까지 폭넓게 커버
def match_item_to_foods(item: str, df_food: pd.DataFrame):
    """item(예: '소고기 미역국')에 대해 food_db의 식품명이 포함되면 매칭.
       반대방향(항목명이 더 짧고 DB가 길 때)도 허용."""
    it = _norm(item)
    hits = df_food[
        df_food["식품"].apply(lambda x: _norm(x) in it or it in _norm(x))
    ].copy()
    # 빈도 줄이기: 완전 불일치 노이즈 제거용 간단 필터
    hits = hits[hits["식품"].apply(lambda x: len(_norm(x)) >= 1)]
    return hits

# --- 분석: 항목별 등급/태그, 영양소 집계
def analyze_diet(input_text: str, df_food: pd.DataFrame, nutrient_desc: dict, threshold:int=1):
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

        # 하나의 입력 항목에 여러 식품이 매칭될 수 있음 → 태그 합산, 등급은 가장 엄격한 것으로
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

    # 영양소 요약 테이블
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
    nutrient_df = pd.DataFrame(rows).sort_values(["상태","횟수","영양소"], ascending=[True, False, True])
    items_df = pd.DataFrame(per_item_rows)
    return items_df, nutrient_df, nutrient_counts

# --- 추천: 부족 영양소 → Safe 식품 추천 (간단/직관)
def recommend_next_meal(nutrient_counts: dict, df_food: pd.DataFrame, nutrient_desc: dict, top_nutrients:int=2, per_food:int=4):
    # 부족 영양소 우선순위
    tag_totals = {t: float(nutrient_counts.get(t, 0)) for t in {tt for lst in df_food["태그리스트"] for tt in lst}}
    lacking = [t for t, v in sorted(tag_totals.items(), key=lambda x: x[1]) if v < 1.0]
    lacking = lacking[:top_nutrients]

    suggestions = []
    for tag in lacking:
        # Safe 위주 추천
        pool = df_food[(df_food["등급"] == "Safe") & (df_food["태그리스트"].apply(lambda lst: tag in lst))]
        foods = pool["식품"].dropna().astype(str).head(per_food).tolist()
        suggestions.append({
            "부족영양소": tag,
            "설명": nutrient_desc.get(tag, ""),
            "추천식품": foods
        })

    # 간단 조합 제안: 부족 1~2개에서 각 1~2개씩 뽑아 3~5개로 구성
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

st.markdown("### 🔎 간단 분석 & 다음 식사 제안")
user_input = st.text_area("식단 텍스트 (쉼표/줄바꿈 구분)", height=90, placeholder="예: 소고기 미역국, 현미밥, 연어구이")

if st.button("분석하기", type="primary"):
    try:
        items_df, nutrient_df, counts = analyze_diet(user_input, FOOD_DB_SIMPLE, NUTRIENT_DESC, threshold=1)

        st.markdown("**🍱 항목별 매칭 결과**")
        if items_df.empty:
            st.info("매칭된 항목이 없습니다.")
        else:
            st.dataframe(items_df, use_container_width=True, height=min(300, 36*(len(items_df)+1)))

        st.markdown("**🧭 영양 태그 요약** (충족/부족 + 한줄설명)")
        if nutrient_df.empty:
            st.info("영양소 사전(nutrient_dict.csv) 또는 태그 정보가 비어 있습니다.")
        else:
            st.dataframe(nutrient_df, use_container_width=True, height=min(360, 36*(len(nutrient_df)+1)))

        st.markdown("**🍽 다음 식사 제안** (부족 영양소 보완)")
        recs, combo = recommend_next_meal(counts, FOOD_DB_SIMPLE, NUTRIENT_DESC, top_nutrients=2, per_food=4)

        if not recs:
            st.success("핵심 부족 영양소가 없습니다. 균형 잘 맞았어요!")
        else:
            for r in recs:
                foods_text = ", ".join(r["추천식품"]) if r["추천식품"] else "(추천 식품 없음)"
                st.write(f"- **{r['부족영양소']}**: {r['설명']}")
                st.caption(f"  추천 식품: {foods_text}")
            if combo:
                st.info("간단 조합 제안: " + " / ".join(combo[:4]))
    except Exception as e:
        st.error(f"분석 중 오류: {e}")

