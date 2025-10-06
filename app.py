
import streamlit as st
import pandas as pd
import json, re, random, time, os
from datetime import date, time as dtime, datetime

st.set_page_config(page_title="민감도 식사 로그 • 개인 규칙 반영", page_icon="🥣", layout="wide")

FOOD_DB_PATH = "food_db.csv"
LOG_PATH = "log.csv"
USER_RULES_PATH = "user_rules.json"

SLOTS = ["오전","오전 간식","점심","오후","오후 간식","저녁"]
EVENT_TYPES = ["food","supplement","symptom","sleep","stool","note"]

CORE_NUTRIENTS = ["Protein","LightProtein","ComplexCarb","HealthyFat","Fiber",
                  "A","B","C","D","E","K","Fe","Mg","Omega3","K_potassium",
                  "Iodine","Ca","Hydration","Circulation"]

ESSENTIALS = ["Protein","ComplexCarb","Fiber","B","C","A","K","Mg","Omega3","K_potassium","HealthyFat","D"]

SUGGEST_MODES = ["기본","달다구리(당김)","역류","더부룩","붓기","피곤함","변비"]

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

# ------- Keyword mappings & virtuals --------
KEYWORD_MAP = {
    "블랙커피": "커피",
    "커피": "커피",
    "녹차": "녹차",
    "홍차": "홍차",
    "사과": "사과",
    "바나나": "바나나",
    "키위": "키위",
    "코코넛 케피어": "__VIRTUAL_COCONUT_KEFIR__",
    "케피어": "__VIRTUAL_COCONUT_KEFIR__",
    "비건치즈": "__VIRTUAL_VEGAN_CHEESE__",
    "베간치즈": "__VIRTUAL_VEGAN_CHEESE__",
    "햄": "__VIRTUAL_HAM__",
    "빵": "__VIRTUAL_BREAD__",
    "현미": "__VIRTUAL_BROWN_RICE__",
    "현미밥": "__VIRTUAL_BROWN_RICE__",
    "brown rice": "__VIRTUAL_BROWN_RICE__",
}

VIRTUAL_RULES = {
    "__VIRTUAL_BREAD__": {"grade":"Avoid","flags":"글루텐/효모 가능성","tags":["ComplexCarb"]},
    "__VIRTUAL_HAM__": {"grade":"Caution","flags":"가공육/염분","tags":["Protein"]},
    "__VIRTUAL_VEGAN_CHEESE__": {"grade":"Caution","flags":"가공 대체식","tags":["HealthyFat"]},
    "__VIRTUAL_COCONUT_KEFIR__": {"grade":"Caution","flags":"발효(프로바이오틱)","tags":["Probiotic"]},
    "__VIRTUAL_BROWN_RICE__": {"grade":"Avoid","flags":"개인 회피: 현미","tags":["ComplexCarb"]},
}

# ---------- Persistence for user rules ----------
def load_user_rules():
    # defaults: avoid brown rice; allow coffee
    defaults = {"avoid_keywords": ["현미","현미밥","brown rice"], "allow_keywords": ["커피"]}
    if os.path.exists(USER_RULES_PATH):
        try:
            with open(USER_RULES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # merge defaults without overwriting user choices
            for k,v in defaults.items():
                if k not in data: data[k]=v
            return data
        except Exception:
            return defaults
    return defaults

def save_user_rules(rules: dict):
    with open(USER_RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

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

# --------- Free-text parsing ---------
def split_free_text(s: str):
    if not s: return []
    extra = []
    for m in re.findall(r"\((.*?)\)", s):
        extra += re.split(r"[,+/ ]+", m)
    s2 = re.sub(r"\(.*?\)", "", s)
    tokens = re.split(r"[+/,]", s2)
    tokens = [t.strip() for t in tokens if t.strip()]
    tokens += [t.strip() for t in extra if t.strip()]
    return tokens

def parse_qty(token: str):
    m = re.search(r"([0-9]+(\.[0-9]+)?)", token)
    qty = float(m.group(1)) if m else 1.0
    name = re.sub(r"[0-9]+(\.[0-9]+)?", "", token).strip()
    name = name.replace("개","").replace("P","").replace("p","").strip()
    return name, qty

def match_food(name: str, food_db: pd.DataFrame):
    orig = name
    if name in KEYWORD_MAP:
        mapped = KEYWORD_MAP[name]
        return mapped, True
    recs = food_db[food_db["식품"]==name]
    if not recs.empty:
        return name, True
    candidates = food_db[food_db["식품"].str.contains(name, case=False, na=False)]
    if not candidates.empty:
        return candidates.iloc[0]["식품"], True
    candidates = food_db[food_db["식품"].apply(lambda x: name in str(x))]
    if not candidates.empty:
        return candidates.iloc[0]["식품"], True
    return orig, False

def contains_any(text, keywords):
    t = str(text)
    for k in keywords:
        if k and k in t:
            return True
    return False

def log_free_foods(log, when_date, when_time, slot, memo, food_db, user_rules):
    tokens = split_free_text(memo)
    saved = []
    for tok in tokens:
        name_raw, qty = parse_qty(tok)
        name_norm = name_raw.strip()
        if not name_norm: continue
        # personal avoid override
        if contains_any(name_norm, user_rules.get("avoid_keywords", [])):
            log = add_log_row(log, when_date, when_time, slot, "food", name_raw, qty, "", "Avoid", "개인 회피리스트", [""], source="memo(personal_avoid)")
            saved.append((name_raw, qty))
            continue

        mapped, matched = match_food(name_norm, food_db)
        if matched:
            if mapped in VIRTUAL_RULES:
                vr = VIRTUAL_RULES[mapped]
                # if this virtual matches allow list (e.g., coffee), downgrade to Safe
                grade = vr["grade"]; flags = vr["flags"]; tags = vr["tags"]
                if contains_any(name_norm, user_rules.get("allow_keywords", [])):
                    grade = "Safe"; flags = "개인 허용"
                log = add_log_row(log, when_date, when_time, slot, "food", name_raw, qty, name_norm, grade, flags, tags, source="memo")
            else:
                rec = food_db[food_db["식품"]==mapped].iloc[0]
                grade = rec.get("등급","Safe")
                tags = rec.get("태그(영양)",[])
                # personal allow can override mode-level avoid, but not DB 'Avoid'
                if contains_any(name_norm, user_rules.get("allow_keywords", [])) and grade!="Avoid":
                    grade = "Safe"
                log = add_log_row(log, when_date, when_time, slot, "food", name_raw, qty, mapped, grade, "", tags, source="memo")
        else:
            # heuristic
            grade, flags, tags = "", "", []
            if "빵" in name_norm: grade, flags = "Avoid", "글루텐/효모 가능성"; tags=["ComplexCarb"]
            if "햄" in name_norm: grade, flags = "Caution", "가공육/염분"; tags=["Protein"]
            if "치즈" in name_norm and ("비건" in name_norm or "베간" in name_norm): grade, flags = "Caution","가공 대체식"; tags=["HealthyFat"]
            if contains_any(name_norm, user_rules.get("allow_keywords", [])): grade="Safe"
            if contains_any(name_norm, user_rules.get("avoid_keywords", [])): grade="Avoid"; flags="개인 회피리스트"
            log = add_log_row(log, when_date, when_time, slot, "food", name_raw, qty, "", grade, flags, tags, source="memo(unmatched)")
        saved.append((name_raw, qty))
    return log, saved

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
            try:
                tags = json.loads(row.get("tags") or "[]")
            except Exception:
                tags = []
            for t in tags:
                if t in score:
                    score[t] += qty
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
    fruits = pool[(pool["식품군"]=="과일")]["식품"].tolist()
    return {"protein":proteins, "veg":vegs, "carb":carbs, "fat":fats, "fruit":fruits}

def mode_filters(mode, user_rules):
    avoid_keywords = []
    comp = {"protein":1,"veg":2,"carb":1,"fat":1,"fruit":0}
    favor = []
    if mode=="기본":
        pass
    elif mode=="달다구리(당김)":
        comp = {"protein":1,"veg":1,"carb":0,"fat":1,"fruit":1}
        avoid_keywords += ["초콜릿","케이크","크림","튀김"]
        favor += ["C","Fiber","K_potassium","HealthyFat"]
    elif mode=="역류":
        avoid_keywords += ["홍차","초콜릿","오렌지","레몬","라임","붉은 고추","스파이시","튀김","크림"]
        # ☕ 커피는 개인 허용 리스트에 있으면 제외하지 않음
        if "커피" not in user_rules.get("allow_keywords", []):
            avoid_keywords += ["커피"]
        comp = {"protein":1,"veg":2,"carb":1,"fat":1}
        favor += ["LightProtein","Fiber"]
    elif mode=="더부룩":
        avoid_keywords += ["양파","마늘","강낭콩","렌틸","완두","콩","브로콜리","양배추","붉은 양배추"]
        comp = {"protein":1,"veg":2,"carb":1,"fat":1}
        favor += ["LightProtein","Fiber"]
    elif mode=="붓기":
        avoid_keywords += ["절임","젓갈","우메보시","김치","햄","베이컨","가공","스톡","간장"]
        comp = {"protein":1,"veg":2,"carb":1,"fat":1,"fruit":0}
        favor += ["K_potassium","Fiber","Hydration"]
    elif mode=="피곤함":
        avoid_keywords += ["튀김","크림","과음"]
        comp = {"protein":1,"veg":2,"carb":1,"fat":1}
        favor += ["B","Fe","Mg","ComplexCarb"]
    elif mode=="변비":
        comp = {"protein":1,"veg":2,"carb":1,"fat":1,"fruit":1}
        avoid_keywords += ["치즈","크림","튀김"]
        favor += ["Fiber","Hydration","K_potassium","HealthyFat"]
    # Apply personal avoids to suggestion filtering later
    return avoid_keywords, comp, favor

def filter_keywords(items, kws):
    return [it for it in items if not any(k in it for k in kws)]

def filter_personal(items, user_rules):
    avoid = user_rules.get("avoid_keywords", [])
    return [it for it in items if not any(k in it for k in avoid)]

def pick_diverse(candidates, recent, need, rng):
    pool = [c for c in candidates if c not in recent]
    if len(pool) >= need:
        rng.shuffle(pool)
        return pool[:need]
    left = need - len(pool)
    rng.shuffle(pool)
    repeat_pool = [c for c in candidates if c in recent]
    rng.shuffle(repeat_pool)
    return pool + repeat_pool[:left]

def gen_meal(df_food, include_caution, mode, recent_items, favor_tags, rng, user_rules):
    baskets = build_baskets(df_food, include_caution=include_caution)
    avoid_kws, comp, favor_extra = mode_filters(mode, user_rules)
    # Apply mode avoid keywords
    for k in list(baskets.keys()):
        baskets[k] = filter_keywords(baskets[k], avoid_kws)
    # Apply personal avoid keywords
    for k in list(baskets.keys()):
        baskets[k] = filter_personal(baskets[k], user_rules)

    local_favor = list(dict.fromkeys(list(favor_tags) + list(favor_extra)))
    def favor(lst):
        if not lst: return lst
        scored = []
        for name in lst:
            tags = df_food[df_food["식품"]==name].iloc[0]["태그(영양)"]
            score = sum(1 for t in local_favor if t in tags)
            scored.append((score, name))
        scored.sort(key=lambda x: (-x[0], rng.random()))
        return [n for _, n in scored]
    for key in baskets.keys():
        baskets[key] = favor(baskets[key])

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
food_db = load_food_db()
log = ensure_log()
user_rules = load_user_rules()

st.title("🥣 민감도 식사 로그 • 개인 회피/허용 반영")

# Sidebar: personal rules editor
with st.sidebar:
    st.subheader("개인 규칙")
    avoid_str = st.text_input("회피 키워드(쉼표 구분)", value=", ".join(user_rules.get("avoid_keywords", [])))
    allow_str = st.text_input("허용 키워드(쉼표 구분)", value=", ".join(user_rules.get("allow_keywords", [])))
    if st.button("규칙 저장"):
        user_rules["avoid_keywords"] = [s.strip() for s in avoid_str.split(",") if s.strip()]
        user_rules["allow_keywords"] = [s.strip() for s in allow_str.split(",") if s.strip()]
        save_user_rules(user_rules)
        st.success("개인 규칙을 저장했어요. (제안/파서에 즉시 반영)")

tab1, tab2, tab3 = st.tabs(["📝 기록","📊 요약/제안","📤 내보내기"])

with tab1:
    st.subheader("오늘 기록")
    d = st.date_input("날짜", value=date.today())
    slot = st.selectbox("슬롯(시간대)", SLOTS, index=2)
    t_input = st.time_input("시각", value=dtime(hour=12, minute=0))
    typ = st.radio("기록 종류", EVENT_TYPES, horizontal=True, index=0)
    if typ=="food":
        memo = st.text_area("메모 한 줄로 입력", height=100, placeholder="예: 빵1, 베간치즈 슬라이스1, 햄1, 블랙커피1, 코코넛 케피어+과일퓨레(사과, 바나나)")
        if st.button("➕ 파싱해서 모두 저장", type="primary"):
            ds = d.strftime("%Y-%m-%d"); ts = t_input.strftime("%H:%M")
            log, saved = log_free_foods(log, ds, ts, slot, memo, food_db, user_rules)
            st.success(f"{len(saved)}개 항목 저장: " + ", ".join([f"{n}×{q}" for n,q in saved]))
    else:
        text = st.text_area("내용 입력", height=80)
        if st.button("➕ 저장", type="primary"):
            ds = d.strftime("%Y-%m-%d"); ts = t_input.strftime("%H:%M")
            log = add_log_row(log, ds, ts, slot, typ, text, 1.0, "", "", "", [], source="manual")
            st.success("저장되었습니다.")
    st.markdown("---")
    st.caption("최근 기록")
    st.dataframe(log.sort_values(['date','time']).tail(20), use_container_width=True, height=240)

with tab2:
    st.subheader("요약 & 다음 끼니 제안(3가지)")
    dsum = st.date_input("기준 날짜", value=date.today(), key="sumdate_2")
    date_str = dsum.strftime("%Y-%m-%d")
    scores = score_day(log, food_db, date_str)
    score_df = pd.DataFrame([scores]).T.reset_index()
    score_df.columns = ["영양소","점수"]
    st.dataframe(score_df, use_container_width=True, height=260)

    favor_tags = [n for n in ESSENTIALS if scores.get(n,0)<1]
    include_caution = st.checkbox("경계(Caution) 포함", value=False)
    diversity_n = st.slider("다양화(최근 N회 중복 회피)", min_value=0, max_value=10, value=5, step=1)
    recent_items = []
    if diversity_n>0:
        recent_df = log[log["type"]=="food"].sort_values(["date","time"]).tail(diversity_n*5)
        recent_items = (recent_df["food_norm"].fillna("") + "|" + recent_df["item"].fillna("")).tolist()
        recent_items = [x.split("|")[0] for x in recent_items if x]

    mode = st.selectbox("제안 모드", SUGGEST_MODES, index=0)
    rng = random.Random(int(time.time()) % 10**9)

    cols = st.columns(3)
    for idx in range(3):
        meal = gen_meal(food_db, include_caution, mode, recent_items, favor_tags, rng, user_rules)
        with cols[idx]:
            st.markdown(f"**제안 {idx+1} — {mode}**")
            if meal:
                st.write("• " + " / ".join(meal))
                if favor_tags:
                    st.caption("부족 보완 우선 태그: " + ", ".join(favor_tags))
                if st.button(f"💾 이 조합 저장 (점심) — {idx+1}"):
                    now = datetime.now().strftime("%H:%M")
                    for token in meal:
                        if token in food_db["식품"].values:
                            rec = food_db[food_db["식품"]==token].iloc[0]
                            add_log_row(log, date_str, now, "점심", "food", token, 1.0, token, rec.get("등급","Safe"), "", rec.get("태그(영양)",[]), source="suggested")
                    st.success("저장 완료! 기록 탭에서 확인하세요.")
            else:
                st.info("조건을 만족하는 식품 풀이 부족합니다. 개인 회피리스트 또는 FoodDB를 확인하세요.")

with tab3:
    st.subheader("내보내기/백업")
    with open(LOG_PATH, "rb") as f:
        st.download_button("⬇️ log.csv 다운로드", data=f, file_name="log.csv", mime="text/csv")
    with open(FOOD_DB_PATH, "rb") as f:
        st.download_button("⬇️ food_db.csv 다운로드", data=f, file_name="food_db.csv", mime="text/csv")
    if os.path.exists(USER_RULES_PATH):
        with open(USER_RULES_PATH, "rb") as f:
            st.download_button("⬇️ user_rules.json 다운로드", data=f, file_name="user_rules.json", mime="application/json")
    st.caption("사이드바에서 개인 회피/허용 키워드를 관리하세요. 예) 회피: 현미, 현미밥 / 허용: 커피")
