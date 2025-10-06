import streamlit as st
import pandas as pd
import json, re, random, time, os, io, zipfile, math
from datetime import date, time as dtime, datetime

# --------------------------------------------------------------------------------
# ⚙️ 기본 설정
# --------------------------------------------------------------------------------
st.set_page_config(page_title="민감도 식사 로그 • 현실형 제안 (안정화)", page_icon="🥣", layout="wide")

FOOD_DB_PATH = "food_db.csv"
NUTRIENT_DICT_PATH = "nutrient_dict.csv"
LOG_PATH = "log.csv"
USER_RULES_PATH = "user_rules.json"

# --------------------------------------------------------------------------------
# 📦 데이터 로드
# --------------------------------------------------------------------------------
@st.cache_data
def load_food_db():
    base_cols = ["식품","식품군","등급","태그(영양)"]
    if not os.path.exists(FOOD_DB_PATH):
        pd.DataFrame(columns=base_cols).to_csv(FOOD_DB_PATH, index=False)
    try:
        df = pd.read_csv(FOOD_DB_PATH, encoding="utf-8", engine="python")
    except Exception:
        df = pd.DataFrame(columns=base_cols)
    if "태그(영양)" in df.columns:
        def safe_json_loads(x):
            if isinstance(x, list): return x
            if pd.isna(x): return []
            s = str(x).strip()
            if s.startswith("[") and "'" in s and '"' not in s:
                s = s.replace("'", '"')
            try:
                return json.loads(s)
            except Exception:
                return [t.strip() for t in s.split(",") if t.strip()]
        df["태그(영양)"] = df["태그(영양)"].apply(safe_json_loads)
    return df

@st.cache_data
def load_nutrient_dict():
    try:
        df = pd.read_csv(NUTRIENT_DICT_PATH)
    except Exception:
        df = pd.DataFrame(columns=["ingredient","protein","fat","carbs","kcal"])
    return df

food_df = load_food_db()
nutrient_df = load_nutrient_dict()

# --------------------------------------------------------------------------------
# 🧮 영양 분석 함수
# --------------------------------------------------------------------------------
def analyze_meal_nutrients(meal_text: str, food_df: pd.DataFrame, nutrient_df: pd.DataFrame):
    tokens = [t.strip() for t in re.split(r"[,+/\n]", meal_text) if t.strip()]
    matched_items = []

    for tok in tokens:
        name = re.sub(r"[0-9]+(\.[0-9]+)?", "", tok).strip()
        if not name:
            continue

        # food_db 매칭
        rec = food_df[food_df["식품"].astype(str).str.contains(name, case=False, na=False)]
        grade = rec.iloc[0]["등급"] if not rec.empty else "N/A"
        tags = rec.iloc[0]["태그(영양)"] if not rec.empty else []

        # nutrient_dict 매칭
        nutrient_info = nutrient_df[nutrient_df["ingredient"].astype(str).str.contains(name, case=False, na=False)]
        if not nutrient_info.empty:
            nutrients = {
                "단백질": float(nutrient_info["protein"].values[0]),
                "지방": float(nutrient_info["fat"].values[0]),
                "탄수화물": float(nutrient_info["carbs"].values[0]),
                "칼로리": float(nutrient_info["kcal"].values[0])
            }
        else:
            nutrients = {"단백질": 0, "지방": 0, "탄수화물": 0, "칼로리": 0}

        matched_items.append({
            "재료": name,
            "등급": grade,
            "태그": tags,
            "영양": nutrients
        })

    # 합산
    total = {
        "단백질": sum([x["영양"]["단백질"] for x in matched_items]),
        "지방": sum([x["영양"]["지방"] for x in matched_items]),
        "탄수화물": sum([x["영양"]["탄수화물"] for x in matched_items]),
        "칼로리": sum([x["영양"]["칼로리"] for x in matched_items])
    }

    nutrient_sum = total["단백질"] + total["지방"] + total["탄수화물"]
    ratio = {k: round(v / nutrient_sum * 100, 1) if nutrient_sum else 0
             for k, v in total.items() if k != "칼로리"}

    return matched_items, total, ratio

# --------------------------------------------------------------------------------
# 🧭 Streamlit UI
# --------------------------------------------------------------------------------
st.title("🥣 민감도 식사 로그 + 영양 분석기 (통합 버전)")

tab1, tab2 = st.tabs(["🍱 영양소 분석", "📘 DB 정보 보기"])

with tab1:
    st.subheader("🔍 식사 영양 분석 (food_db + nutrient_dict 연동)")
    st.caption("입력된 음식명에 포함된 재료를 food_db와 nutrient_dict에서 찾아 영양 정보를 계산합니다.")

    sample_text = "소고기 미역국"
    meal_input = st.text_input("식사명 입력", value=sample_text, placeholder="예: 소고기 미역국, 닭가슴살 샐러드")

    if st.button("분석 실행", type="primary"):
        if not meal_input.strip():
            st.warning("음식명을 입력해주세요.")
        else:
            items, total, ratio = analyze_meal_nutrients(meal_input, food_df, nutrient_df)

            if not items:
                st.info("매칭된 재료가 없습니다.")
            else:
                st.markdown("### 🍲 매칭된 재료 및 영양소")
                st.dataframe(
                    pd.DataFrame([
                        {
                            "재료": i["재료"],
                            "등급": i["등급"],
                            "태그": ", ".join(i["태그"]) if isinstance(i["태그"], list) else i["태그"],
                            **i["영양"]
                        } for i in items
                    ]),
                    use_container_width=True
                )

                st.markdown("### 📊 총 영양 성분")
                st.json(total)

                st.markdown("### ⚖️ 영양 비율 (%)")
                st.bar_chart(pd.DataFrame(ratio, index=["비율"]).T)

                # 추가 분석: 부족/과다 포인트
                st.markdown("### 🩺 분석 요약")
                protein = total["단백질"]
                fat = total["지방"]
                carb = total["탄수화물"]

                msg = []
                if protein < 15:
                    msg.append("단백질이 부족할 수 있습니다. (두부·계란·생선 추가 권장)")
                if fat > 30:
                    msg.append("지방이 다소 높습니다. 조리유를 줄여보세요.")
                if carb > 60:
                    msg.append("탄수화물 비율이 높습니다. 섬유질 식품을 보완해보세요.")
                if not msg:
                    msg.append("균형 잡힌 식사로 보입니다 ✅")
                st.write("\n".join([f"- {m}" for m in msg]))

with tab2:
    st.subheader("📘 현재 로드된 데이터베이스 정보")
    st.write(f"food_db.csv: {len(food_df)}개 항목")
    st.write(food_df.head(10))
    st.write(f"nutrient_dict.csv: {len(nutrient_df)}개 항목")
    st.write(nutrient_df.head(10))

st.caption("© 2025 식사 민감도 분석 및 영양 통합 버전 — 자동 매칭 및 영양 계산 포함")
