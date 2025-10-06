# ✅ Streamlit 민감도 식사 로그 (안정화 버전)
# FoodDB UI 제거 + 희귀 식재료 자동 제외 + 상태별 제안 확실히 구분
import streamlit as st, pandas as pd, json, os, re, random, time
from datetime import date, datetime

st.set_page_config(page_title="민감도 식사 로그", page_icon="🥣", layout="wide")

LOG_PATH = "log.csv"
USER_RULES_PATH = "user_rules.json"

# ---------------- 기본설정 ----------------
SUGGEST_MODES = ["기본", "달다구리(당김)", "역류", "더부룩", "붓기", "피곤함", "변비"]
COMMON_WHITELIST = {
    "protein": ["닭가슴살","대구","연어","돼지고기","소고기","고등어","참치(캔)"],
    "veg": ["양배추","당근","브로콜리","애호박","오이","시금치","상추","무","감자","고구마","파프리카","토마토"],
    "carb": ["쌀밥","쌀죽","고구마","감자","퀴노아","타피오카"],
    "fat": ["올리브유","들기름","참기름","아보카도","참깨"],
    "fruit": ["사과","바나나","키위","블루베리","딸기","배"]
}

MODE_ANCHORS = {
    "기본": {"protein":["닭가슴살","연어","대구"],"veg":["브로콜리","시금치","당근"],"carb":["쌀밥","고구마"],"fat":["올리브유"],"fruit":["사과","키위"]},
    "달다구리(당김)": {"protein":["닭가슴살"],"veg":["시금치","오이"],"carb":["퀴노아","타피오카"],"fat":["올리브유"],"fruit":["블루베리","딸기"]},
    "역류": {"protein":["대구","닭가슴살"],"veg":["오이","애호박","시금치"],"carb":["쌀죽","감자"],"fat":["올리브유"],"fruit":["바나나","사과"]},
    "더부룩": {"protein":["대구","닭가슴살"],"veg":["오이","시금치","당근"],"carb":["쌀밥","감자"],"fat":["올리브유"],"fruit":["바나나","키위"]},
    "붓기": {"protein":["연어","닭가슴살"],"veg":["오이","시금치","당근"],"carb":["고구마","퀴노아"],"fat":["올리브유"],"fruit":["바나나","키위"]},
    "피곤함": {"protein":["소고기","돼지고기","연어"],"veg":["시금치","브로콜리"],"carb":["고구마","퀴노아"],"fat":["올리브유"],"fruit":["키위","사과"]},
    "변비": {"protein":["연어","닭가슴살"],"veg":["양배추","브로콜리","당근"],"carb":["퀴노아","쌀밥"],"fat":["올리브유","참깨"],"fruit":["키위","사과"]}
}

def load_rules():
    if os.path.exists(USER_RULES_PATH):
        try:
            return json.load(open(USER_RULES_PATH, encoding="utf-8"))
        except: pass
    return {"avoid_keywords":["현미","현미밥"],"allow_keywords":["커피"]}

def save_rules(rules):
    json.dump(rules, open(USER_RULES_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def ensure_log():
    cols = ["date","time","slot","item","note"]
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=cols).to_csv(LOG_PATH,index=False)
    return pd.read_csv(LOG_PATH)

def add_log(date_str,t_str,slot,item,note):
    df=ensure_log()
    df.loc[len(df)] = [date_str,t_str,slot,item,note]
    df.to_csv(LOG_PATH,index=False)

# ---------------- 제안 생성 ----------------
def suggest_meal(mode:str, allow_rare=False):
    base = MODE_ANCHORS.get(mode, MODE_ANCHORS["기본"])
    out = {}
    for k,v in base.items():
        cand = v.copy()
        if not allow_rare:
            cand = [x for x in cand if x in COMMON_WHITELIST.get(k,[])]
        if cand: out[k]=random.choice(cand)
    txt = " / ".join(out.values())
    return txt if txt else "데이터 부족"

# ---------------- UI ----------------
st.title("🥣 민감도 식사 로그 & 제안기")
rules = load_rules()

tab1,tab2 = st.tabs(["📝 기록","💡 식사 제안"])

with tab1:
    st.subheader("새 기록 추가")
    d = st.date_input("날짜", value=date.today())
    t = st.time_input("시간")
    slot = st.selectbox("식사 구분", ["오전","점심","저녁","간식"])
    item = st.text_input("내용 (예: 빵1, 햄1, 커피1)")
    note = st.text_area("메모", "")
    if st.button("저장"):
        add_log(str(d), str(t), slot, item, note)
        st.success("저장됨 ✅")
    st.divider()
    df = ensure_log()
    st.dataframe(df.tail(10))

with tab2:
    st.subheader("식사 제안")
    mode = st.selectbox("컨디션 모드", SUGGEST_MODES)
    allow_rare = st.checkbox("희귀 재료 포함", value=False)
    if st.button("🔄 새로고침"):
        st.session_state.rand = random.randint(1,9999)
    st.write("제안 식단:")
    st.info(suggest_meal(mode, allow_rare))

st.sidebar.markdown("### ⚙️ 개인 규칙")
avoid = st.text_area("회피 재료(쉼표)", ",".join(rules.get("avoid_keywords",[])))
allow = st.text_area("허용 재료(쉼표)", ",".join(rules.get("allow_keywords",[])))
if st.sidebar.button("규칙 저장"):
    save_rules({"avoid_keywords":[a.strip() for a in avoid.split(",") if a.strip()],
                "allow_keywords":[a.strip() for a in allow.split(",") if a.strip()]})
    st.sidebar.success("저장 완료")
