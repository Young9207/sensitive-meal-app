
import streamlit as st
import pandas as pd
import json
from datetime import datetime, date

st.set_page_config(page_title="민감도 식사 분석", page_icon="🥣", layout="wide")

FOOD_DB_PATH = "food_db.csv"
LOG_PATH = "meals_log.csv"

CORE_NUTRIENTS = ["Protein","LightProtein","ComplexCarb","HealthyFat","Fiber",
                  "A","B","C","D","E","K","Fe","Mg","Omega3","K_potassium",
                  "Iodine","Ca","Hydration","Circulation"]

ESSENTIALS = ["Protein","ComplexCarb","Fiber","B","C","A","K","Mg","Omega3","K_potassium","HealthyFat","D"]

GAP_MAP = {
    "Protein": "대구 100g 또는 치킨 100g",
    "ComplexCarb": "현미 1/2공기 또는 고구마 1/2개",
    "Fiber": "브로콜리 1/2컵 또는 양배추 1/2컵",
    "B": "귀리 1/2공기 또는 현미 소량",
    "C": "사과 1/2개 또는 키위",
    "A": "단호박 1/2컵 또는 당근 1/2개",
    "K": "시금치/브로콜리 1/2컵",
    "Mg": "아보카도 소량 또는 아몬드 5알",
    "Omega3": "연어 80g 또는 고등어 80g",
    "K_potassium": "단호박 1/2컵 또는 미역국 반그릇",
    "HealthyFat": "들기름 1작은술 또는 올리브유 1작은술",
    "D": "연어 80g 또는 햇빛 20분"
}

def load_food_db():
    df = pd.read_csv(FOOD_DB_PATH)
    # ensure tags parsed
    def parse_tags(x):
        try:
            return json.loads(x)
        except Exception:
            return [t.strip() for t in str(x).split(",") if t.strip()]
    df["태그(영양)"] = df["태그(영양)"].apply(parse_tags)
    if "민감_제외_권장" not in df.columns:
        df["민감_제외_권장"] = False
    return df

def init_log():
    cols = ["date","meal","food","servings"]
    try:
        log = pd.read_csv(LOG_PATH)
        # basic schema guard
        for c in cols:
            if c not in log.columns:
                raise ValueError
        return log
    except Exception:
        log = pd.DataFrame(columns=cols)
        log.to_csv(LOG_PATH, index=False)
        return log

def score_day(df_log, df_food, date_str):
    day = df_log[df_log["date"]==date_str]
    score = {k:0 for k in CORE_NUTRIENTS}
    for _, row in day.iterrows():
        food = row["food"]
        servings = float(row.get("servings", 1.0) or 1.0)
        recs = df_food[df_food["식품"]==food]
        if recs.empty:
            continue
        tags = recs.iloc[0]["태그(영양)"]
        for t in tags:
            if t in score:
                score[t] += servings
    return score

def suggest_next_meal(scores, sodium_heavy=False, symptom=None):
    low = [n for n in ESSENTIALS if scores.get(n,0) < 1]
    suggestions = []

    if symptom in ("복부팽만","메스꺼움"):
        suggestions.append(("저자극 회복식",
            ["쌀 죽 1그릇","대구 100g 찜","애호박 1컵(데침)","보리차 1잔"],
            ["Protein","LightProtein","ComplexCarb","Fiber","Hydration"]))
    elif symptom in ("오른쪽윗배묵직","몸무거움") or sodium_heavy:
        suggestions.append(("간·순환 보조(저염)",
            ["귀리죽 또는 현미죽 1그릇",
             "연어 80~100g(구이/찜) 또는 대구 100g",
             "브로콜리 1컵 + 단호박 1/2~1컵",
             "들기름 또는 올리브유 1작은술",
             "루이보스티 또는 보리차"],
            ["A","B","C","K","Omega3","D","HealthyFat","Hydration"]))
    elif symptom == "피로":
        suggestions.append(("피로·집중 보완",
            ["현미 1/2공기","치킨(안심) 100g",
             "시금치 1/2접시 + 사과 1/2개","올리브유 1작은술"],
            ["Protein","B","Fe","Mg","C","HealthyFat"]))
    else:
        suggestions.append(("균형 한 끼",
            ["치킨(안심) 100g 또는 대구 100g",
             "시금치/양배추/브로콜리 중 2가지",
             "현미 1/2공기","올리브유 1작은술","사과 1/2개"],
            ["Protein","B","C","K","HealthyFat","Fiber"]))

    gap_sides = [GAP_MAP[g] for g in low if g in GAP_MAP][:3]
    if gap_sides:
        suggestions.append(("부족 영양 보완 사이드", gap_sides, low))
    return suggestions, low

# -------------------- UI --------------------
st.title("🥣 민감도 식사 분석 (Streamlit)")

food_db = load_food_db()
log = init_log()

tabs = st.tabs(["📥 식사 기록","📊 오늘 요약","🧘 컨디션 & 다음 끼니","📚 FoodDB 편집"])

with tabs[0]:
    st.subheader("📥 식사 기록")
    today = st.date_input("날짜", value=date.today())
    meal = st.selectbox("끼니", ["아침","점심","저녁","간식"])
    # Allow include/exclude sensitive foods
    show_sensitives = st.checkbox("민감_제외_권장 식품 포함해서 보기", value=False)
    df_view = food_db if show_sensitives else food_db[food_db["민감_제외_권장"]==False]
    # group by 식품군 for nicer UI
    group = st.selectbox("식품군", sorted(df_view["식품군"].unique()))
    choices = df_view[df_view["식품군"]==group]["식품"].sort_values().tolist()
    picked = st.multiselect("식품 선택 (여러 개 가능)", choices)
    servings = st.number_input("분량 배수(기본 1.0)", min_value=0.25, max_value=3.0, value=1.0, step=0.25)

    if st.button("로그에 추가"):
        for p in picked:
            log = pd.concat([log, pd.DataFrame([{
                "date": today.strftime("%Y-%m-%d"),
                "meal": meal,
                "food": p,
                "servings": servings
            }])], ignore_index=True)
        log.to_csv(LOG_PATH, index=False)
        st.success(f"{len(picked)}개 항목을 저장했습니다.")
        st.dataframe(log.tail(10), use_container_width=True)

with tabs[1]:
    st.subheader("📊 오늘 영양 점수")
    today = st.date_input("요약할 날짜", value=date.today(), key="sumdate")
    date_str = today.strftime("%Y-%m-%d")
    day_log = log[log["date"]==date_str]
    st.write("오늘 로그", day_log)

    scores = score_day(log, food_db, date_str)
    score_df = pd.DataFrame([scores]).T.reset_index()
    score_df.columns = ["영양소","점수"]
    st.dataframe(score_df, use_container_width=True)

    # sodium heavy heuristic
    sodium_heavy = any(x in ["총각김치","깻잎절임","우메보시"] for x in day_log["food"].tolist())
    st.caption(f"염분 높은 절임/발효식 포함 여부: {'예' if sodium_heavy else '아니요'}")

with tabs[2]:
    st.subheader("🧘 컨디션 체크 & 다음 끼니 제안")
    col1, col2 = st.columns(2)
    with col1:
        symptom = st.selectbox("지금 상태", ["없음","복부팽만","메스꺼움","몸무거움","오른쪽윗배묵직","피로"])
    with col2:
        date_sel = st.date_input("기준 날짜", value=date.today(), key="symdate")
        dstr = date_sel.strftime("%Y-%m-%d")
        s_today = score_day(log, food_db, dstr)
        sodium_heavy = any(x in ["총각김치","깻잎절임","우메보시"] for x in log[log["date"]==dstr]["food"].tolist())

    sug, gaps = suggest_next_meal(s_today, sodium_heavy=sodium_heavy, symptom=(None if symptom=="없음" else symptom))
    for (title, items, targets) in sug:
        st.markdown(f"**{title}**")
        st.write("• " + " / ".join(items))
        if isinstance(targets, list):
            st.caption("보완 대상 영양소: " + ", ".join(targets))

with tabs[3]:
    st.subheader("📚 FoodDB 편집")
    st.write("CSV를 직접 수정하거나 여기에서 추가/수정 후 저장하세요.")
    st.dataframe(food_db, use_container_width=True)
    with st.expander("항목 추가"):
        new_food = st.text_input("식품명")
        new_group = st.text_input("식품군 (예: 채소, 곡물류 등)")
        new_serv = st.text_input("1회분 (예: 1컵)")
        new_tags = st.text_input("태그(쉼표 구분, 예: Protein,B,C)")
        new_avoid = st.checkbox("민감_제외_권장", value=False)
        if st.button("추가/저장"):
            tags = [t.strip() for t in new_tags.split(",") if t.strip()]
            new_row = {"식품":new_food,"식품군":new_group,"1회분":new_serv,"태그(영양)":json.dumps(tags, ensure_ascii=False),"민감_제외_권장":new_avoid}
            food_db = pd.concat([food_db, pd.DataFrame([new_row])], ignore_index=True)
            food_db.to_csv(FOOD_DB_PATH, index=False)
            st.success("저장 완료! 앱을 새로고침하세요.")

st.sidebar.info("Tip: 매 끼니는 '채소 1/2 + 단백질 1/4 + 복합탄수 1/4 + 좋은 지방 소량'을 기본으로 잡으세요.")
