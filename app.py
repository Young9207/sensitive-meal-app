
import streamlit as st
import pandas as pd
import json, random, time
from datetime import date, time as dtime, datetime

st.set_page_config(page_title="민감도 식사 로그 • 제안 강화", page_icon="🥣", layout="wide")

FOOD_DB_PATH = "food_db.csv"
LOG_PATH = "log.csv"

SLOTS = ["오전","오전 간식","점심","오후","오후 간식","저녁"]
EVENT_TYPES = ["food","supplement","symptom","sleep","stool","note"]

CORE_NUTRIENTS = ["Protein","LightProtein","ComplexCarb","HealthyFat","Fiber",
                  "A","B","C","D","E","K","Fe","Mg","Omega3","K_potassium",
                  "Iodine","Ca","Hydration","Circulation"]

ESSENTIALS = ["Protein","ComplexCarb","Fiber","B","C","A","K","Mg","Omega3","K_potassium","HealthyFat","D"]

SUGGEST_MODES = ["기본","저자극(역류/메스꺼움)","저염(붓기/절임 후)","샐러드","죽","외식용"]

SUPP_ALERT_KEYWORDS = {
    "효모": ("Avoid","효모 반응 ↑: 빵/맥주/맥주효모 주의"),
    "맥주효모": ("Avoid","효모 반응 ↑: 맥주효모 회피 권장"),
    "카제인": ("Avoid","유제품·카제인 반응 ↑"),
    "유청": ("Avoid","유제품계 단백(유청) 주의"),
    "whey": ("Avoid","유제품계 단백(유청) 주의"),
    "casein": ("Avoid","유제품·카제인 반응 ↑"),
    "gluten": ("Avoid","글루텐 회피 권장(글리아딘 반응 ↑)"),
    "corn": ("Caution","옥수수 경계: 폴렌타/콘가공품 주의"),
}

# ---------- I/O ----------
def ensure_log():
    cols = ["date","weekday","time","slot","type","item","qty","food_norm","grade","flags","tags","source"]
    try:
        log = pd.read_csv(LOG_PATH)
        for c in cols:
            if c not in log.columns:
                log[c] = "" if c not in ["qty"] else 0
        log = log[cols]
        return log
    except Exception:
        log = pd.DataFrame(columns=cols)
        log.to_csv(LOG_PATH, index=False)
        return log

def load_food_db():
    df = pd.read_csv(FOOD_DB_PATH, encoding="utf-8", engine="python")
    if "태그(영양)" in df.columns:
        def parse_tags(x):
            try:
                return json.loads(x)
            except Exception:
                return [t.strip() for t in str(x).split(",") if t.strip()]
        df["태그(영양)"] = df["태그(영양)"].apply(parse_tags)
    else:
        df["태그(영양)"] = [[] for _ in range(len(df))]
    if "민감_제외_권장" not in df.columns:
        df["민감_제외_권장"] = False
    if "등급" not in df.columns:
        df["등급"] = df["민감_제외_권장"].apply(lambda x: "Caution" if x else "Safe")
    if "식품" not in df.columns and "항목_한글" in df.columns:
        df = df.rename(columns={"항목_한글":"식품"})
    if "식품군" not in df.columns:
        df["식품군"] = ""
    return df

def weekday_ko(dt: date):
    return ["MON","TUE","WED","THU","FRI","SAT","SUN"][dt.weekday()]

def add_log_row(log, date_str, t_str, slot, typ, item, qty, food_norm, grade, flags, tags, source="manual"):
    new = pd.DataFrame([{
        "date": date_str,
        "weekday": weekday_ko(datetime.strptime(date_str,"%Y-%m-%d").date()),
        "time": t_str,
        "slot": slot,
        "type": typ,
        "item": item,
        "qty": qty,
        "food_norm": food_norm,
        "grade": grade,
        "flags": flags,
        "tags": json.dumps(tags, ensure_ascii=False) if isinstance(tags, list) else (tags or ""),
        "source": source
    }])
    log = pd.concat([log, new], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)
    return log

# ---------- Scoring ----------
def score_day(df_log, df_food, date_str):
    day = df_log[(df_log["date"]==date_str) & (df_log["type"]=="food")]
    score = {k:0.0 for k in CORE_NUTRIENTS}
    for _, row in day.iterrows():
        fn = row.get("food_norm") or row.get("item")
        try:
            qty = float(row.get("qty") or 1.0)
        except Exception:
            qty = 1.0
        recs = df_food[df_food["식품"]==fn]
        if recs.empty:
            continue
        tags = recs.iloc[0]["태그(영양)"]
        for t in tags:
            if t in score:
                score[t] += qty
    return score

# ---------- Suggestion engine ----------
def build_baskets(df, include_caution=False):
    pool = df.copy()
    if not include_caution:
        pool = pool[pool["등급"]=="Safe"]
    else:
        pool = pool[pool["등급"].isin(["Safe","Caution"])]
    proteins = pool[(pool["식품군"].isin(["생선/해산물","육류"])) & (pool["태그(영양)"].apply(lambda t: "Protein" in t))]["식품"].tolist()
    vegs = pool[(pool["식품군"]=="채소") & (pool["태그(영양)"].apply(lambda t: "Fiber" in t))]["식품"].tolist()
    carbs = pool[(pool["태그(영양)"].apply(lambda t: "ComplexCarb" in t))]["식품"].tolist()
    fats = pool[(pool["태그(영양)"].apply(lambda t: "HealthyFat" in t))]["식품"].tolist()
    return {"protein":proteins, "veg":vegs, "carb":carbs, "fat":fats}

def mode_filters(mode):
    avoid_keywords = []
    comp = {"protein":1,"veg":2,"carb":1,"fat":1}
    if mode=="저자극(역류/메스꺼움)":
        avoid_keywords += ["커피","홍차","초콜릿","오렌지","레몬","라임","붉은 고추","스파이시"]
    if mode=="저염(붓기/절임 후)":
        avoid_keywords += ["절임","젓갈","우메보시","김치"]
    if mode=="샐러드":
        comp = {"protein":1,"veg":2,"fat":1}  # no carb
    if mode=="죽":
        comp = {"protein":1,"veg":1,"carb":1,"fat":1}
    if mode=="외식용":
        # keep comp; just avoid tricky items
        avoid_keywords += ["튀김","프라이","크림"]
    return avoid_keywords, comp

def filter_keywords(items, kws):
    res = []
    for it in items:
        if any(k in it for k in kws):
            continue
        res.append(it)
    return res

def pick_diverse(candidates, recent, need, rng):
    # avoid items in recent; if insufficient, allow but minimize repeats
    pool = [c for c in candidates if c not in recent]
    if len(pool) >= need:
        rng.shuffle(pool)
        return pool[:need]
    # fallback
    left = need - len(pool)
    rng.shuffle(pool)
    repeat_pool = [c for c in candidates if c in recent]
    rng.shuffle(repeat_pool)
    return pool + repeat_pool[:left]

def gen_meal(df_food, include_caution, mode, recent_items, favor_tags, rng):
    baskets = build_baskets(df_food, include_caution=include_caution)
    avoid_kws, comp = mode_filters(mode)
    # apply keyword filters
    for k in list(baskets.keys()):
        baskets[k] = filter_keywords(baskets[k], avoid_kws)

    # add simple nutrient favoring by reordering candidates
    def favor(lst, tag):
        if not lst: return lst
        scored = []
        for name in lst:
            tags = df_food[df_food["식품"]==name].iloc[0]["태그(영양)"]
            score = sum(1 for t in favor_tags if t in tags)
            scored.append((score, name))
        scored.sort(key=lambda x: (-x[0], rng.random()))
        return [n for _, n in scored]

    baskets["protein"] = favor(baskets["protein"], favor_tags)
    baskets["veg"] = favor(baskets["veg"], favor_tags)
    baskets["carb"] = favor(baskets["carb"], favor_tags)
    baskets["fat"] = favor(baskets["fat"], favor_tags)

    meal = []
    for key, need in comp.items():
        chosen = pick_diverse(baskets[key], recent_items, need, rng)
        meal += chosen
    return meal

def supplement_flag(text):
    if not text: return ("","")
    t = text.lower()
    for key, (grade,msg) in SUPP_ALERT_KEYWORDS.items():
        if key in t:
            return (grade, msg)
    return ("","")

# ---------- App ----------
# auto-refresh for live updates
st_autorefresh = st.sidebar.checkbox("자동 새로고침(15초)", value=False)
if st_autorefresh:
    st.runtime.legacy_caching.clear_cache()
    st.experimental_rerun  # placeholder (Streamlit handles rerun on each call)
    st.autorefresh = st.experimental_singleton(lambda: None)
    st.experimental_rerun

food_db = load_food_db()
log = ensure_log()

st.title("🥣 민감도 식사 로그 • 제안 강화")

tab1, tab2, tab3 = st.tabs(["📝 기록","📊 요약/제안","📤 내보내기"])

with tab1:
    colL, colR = st.columns([2,1])
    with colL:
        st.subheader("오늘 기록")
        d = st.date_input("날짜", value=date.today())
        slot = st.selectbox("슬롯(시간대)", SLOTS, index=2)
        t = st.time_input("시각", value=dtime(hour=12, minute=0))
        typ = st.radio("기록 종류", EVENT_TYPES, horizontal=True, index=0)
        qty = st.number_input("분량/개수(가능시)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        show_sens = st.checkbox("경계/회피 식품도 목록에 보여주기", value=False)

        food_norm=""; grade=""; flags=""; tags=[]

        if typ=="food":
            df_view = food_db.copy()
            if not show_sens:
                df_view = df_view[df_view["등급"]=="Safe"]
            group = st.selectbox("식품군", ["(전체)"] + sorted(df_view["식품군"].dropna().unique().tolist()))
            if group != "(전체)":
                df_view = df_view[df_view["식품군"]==group]
            query = st.text_input("검색(식품명 일부)", value="")
            if query.strip():
                df_view = df_view[df_view["식품"].str.contains(query.strip(), case=False, na=False)]
            food_norm = st.selectbox("식품 선택", [""] + sorted(df_view["식품"].tolist()))
            item = st.text_input("메모(예: 사과1 / 요거트·시나몬)", value="")

            if food_norm:
                rec = food_db[food_db["식품"]==food_norm].iloc[0]
                grade = rec.get("등급","Safe")
                tags = rec.get("태그(영양)",[])
                badge = "🟢 Safe" if grade=="Safe" else ("🟡 Caution" if grade=="Caution" else "🔴 Avoid")
                st.markdown(f"**등급:** {badge}  •  **태그:** {', '.join(tags) if tags else '-'}")
                if grade=="Avoid":
                    st.error("검사 기준: 회피 권장 항목입니다.")
                elif grade=="Caution":
                    st.warning("검사 기준: 경계 항목입니다. 순환/소량 권장.")

            if st.button("➕ 로그 저장", type="primary"):
                date_str = d.strftime("%Y-%m-%d")
                t_str = t.strftime("%H:%M")
                log = add_log_row(log, date_str, t_str, slot, typ, item, qty, food_norm, grade, flags, tags, source="manual")
                st.success("저장되었습니다.")
        else:
            item = st.text_input("내용 입력", value="")
            if typ=="supplement":
                g, flags = supplement_flag(item)
                grade = g
                if g=="Avoid":
                    st.error(flags or "주의 보충제")
                elif g=="Caution":
                    st.warning(flags or "경계 보충제")
            if st.button("➕ 로그 저장", type="primary"):
                date_str = d.strftime("%Y-%m-%d")
                t_str = t.strftime("%H:%M")
                log = add_log_row(log, date_str, t_str, slot, typ, item, qty, "", grade, flags, [], source="manual")
                st.success("저장되었습니다.")

        st.markdown("---")
        st.caption("최근 기록")
        st.dataframe(log.sort_values(["date","time"]).tail(20), use_container_width=True, height=240)

    with colR:
        st.subheader("오늘 영양 점수")
        dsum = st.date_input("요약 날짜", value=date.today(), key="sumdate_1")
        date_str = dsum.strftime("%Y-%m-%d")
        day = log[log["date"]==date_str]
        sodium_heavy = any(k in (day["item"].fillna("") + " " + day["food_norm"].fillna("")).str.cat(sep=" ")
                           for k in ["김치","우메보시","절임","장아찌","젓갈"])
        scores = score_day(log, food_db, date_str)
        score_df = pd.DataFrame([scores]).T.reset_index()
        score_df.columns = ["영양소","점수"]
        st.dataframe(score_df, use_container_width=True, height=300)

with tab2:
    st.subheader("다음 끼니 제안(3가지)")
    dsum = st.date_input("기준 날짜", value=date.today(), key="sumdate_2")
    date_str = dsum.strftime("%Y-%m-%d")
    scores = score_day(log, food_db, date_str)
    # 부족 태그 계산
    favor_tags = [n for n in ESSENTIALS if scores.get(n,0)<1]
    # 증상 수집
    day = log[log["date"]==date_str]
    sym_today = day[day["type"]=="symptom"]["item"].str.cat(sep=" ").lower()
    symptoms = []
    for key in ["역류","신물","메스꺼움","복부팽만","붓기","피로"]:
        if key in sym_today:
            symptoms.append(key)
    # 모드 & 옵션
    mode = st.selectbox("제안 모드", SUGGEST_MODES, index=0)
    include_caution = st.checkbox("경계(Caution) 포함", value=False)
    diversity_n = st.slider("다양화(최근 N회 중복 회피)", min_value=0, max_value=10, value=5, step=1)
    # 최근 N회 재료 수집
    recent_items = []
    if diversity_n>0:
        recent_df = log[log["type"]=="food"].sort_values(["date","time"]).tail(diversity_n*5)
        recent_items = (recent_df["food_norm"].fillna("") + "|" + recent_df["item"].fillna("")).tolist()
        # normalize to just food_norm names if exist
        recent_items = [x.split("|")[0] for x in recent_items if x]
    # RNG seed 버튼
    if "seed" not in st.session_state:
        st.session_state.seed = int(time.time())
    if st.button("🔀 다른 조합 보기"):
        st.session_state.seed = random.randint(1, 10**9)
    rng = random.Random(st.session_state.seed)

    # sodium mode auto if needed
    if any(k in symptoms for k in ["붓기"]) or any(term in sym_today for term in ["절임","젓갈","우메보시","김치"]):
        if mode=="기본":
            mode = "저염(붓기/절임 후)"

    # 모드가 저자극이면 favor_tags에서 산성 쪽은 제외 가중(간단)
    if mode=="저자극(역류/메스꺼움)":
        # not implementing complex reweight; keywords handled in mode_filters
        pass

    # 3가지 제안 생성
    cols = st.columns(3)
    for idx in range(3):
        meal = gen_meal(food_db, include_caution, mode, recent_items, favor_tags, rng)
        with cols[idx]:
            st.markdown(f"**제안 {idx+1} — {mode}**")
            if meal:
                st.write("• " + " / ".join(meal))
                if favor_tags:
                    st.caption("부족 보완 우선 태그: " + ", ".join(favor_tags))
                # 저장 버튼
                if st.button(f"💾 이 조합 저장 (점심) — {idx+1}"):
                    now = datetime.now().strftime("%H:%M")
                    for token in meal:
                        if token in food_db["식품"].values:
                            rec = food_db[food_db["식품"]==token].iloc[0]
                            add_log_row(log, date_str, now, "점심", "food", token, 1.0, token, rec.get("등급","Safe"), "", rec.get("태그(영양)",[]), source="suggested")
                    st.success("저장 완료! 기록 탭에서 확인하세요.")
            else:
                st.info("조건을 만족하는 식품 풀이 부족합니다. FoodDB를 보강해주세요.")

with tab3:
    st.subheader("내보내기/백업")
    # Download buttons
    with open(LOG_PATH, "rb") as f:
        st.download_button("⬇️ log.csv 다운로드", data=f, file_name="log.csv", mime="text/csv")
    with open(FOOD_DB_PATH, "rb") as f:
        st.download_button("⬇️ food_db.csv 다운로드", data=f, file_name="food_db.csv", mime="text/csv")

    st.markdown("---")
    st.caption("구글 드라이브에 업로드하려면 아래에 폴더(또는 MyDrive) 공유 링크를 붙여두고, 다운로드한 파일을 수동 업로드하세요.")
    drive_url = st.text_input("내 구글 드라이브 폴더 링크(선택)", value="", help="예: https://drive.google.com/drive/folders/....")
    if drive_url.strip():
        st.markdown(f"[🟢 구글 드라이브 폴더 열기]({drive_url})")

st.sidebar.info("제안 모드/경계 포함/다양화로 여러 조합을 만들어 보세요.")
