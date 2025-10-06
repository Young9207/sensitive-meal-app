
import streamlit as st
import pandas as pd
import json, re, random, time, os, io, zipfile
from datetime import date, time as dtime, datetime

st.set_page_config(page_title="민감도 식사 로그 • 현실형 제안", page_icon="🥣", layout="wide")

FOOD_DB_PATH = "food_db.csv"
LOG_PATH = "log.csv"
USER_RULES_PATH = "user_rules.json"

SLOTS = ["오전","오전 간식","점심","오후","오후 간식","저녁"]
EVENT_TYPES = ["food","supplement","symptom"]  # 단순화

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

# ------- 자유입력 키워드 & 가상 매핑 --------
KEYWORD_MAP = {
    "블랙커피": "커피", "커피": "커피",
    "녹차": "녹차", "홍차": "홍차",
    "사과": "사과", "바나나": "바나나", "키위": "키위",
    "코코넛 케피어": "__VIRTUAL_COCONUT_KEFIR__", "케피어": "__VIRTUAL_COCONUT_KEFIR__",
    "비건치즈": "__VIRTUAL_VEGAN_CHEESE__", "베간치즈": "__VIRTUAL_VEGAN_CHEESE__",
    "햄": "__VIRTUAL_HAM__", "빵": "__VIRTUAL_BREAD__",
    "현미": "__VIRTUAL_BROWN_RICE__", "현미밥": "__VIRTUAL_BROWN_RICE__", "brown rice": "__VIRTUAL_BROWN_RICE__",
}

VIRTUAL_RULES = {
    "__VIRTUAL_BREAD__": {"grade":"Avoid","flags":"글루텐/효모 가능성","tags":["ComplexCarb"]},
    "__VIRTUAL_HAM__": {"grade":"Caution","flags":"가공육/염분","tags":["Protein"]},
    "__VIRTUAL_VEGAN_CHEESE__": {"grade":"Caution","flags":"가공 대체식","tags":["HealthyFat"]},
    "__VIRTUAL_COCONUT_KEFIR__": {"grade":"Caution","flags":"발효(프로바이오틱)","tags":["Probiotic"]},
    "__VIRTUAL_BROWN_RICE__": {"grade":"Avoid","flags":"개인 회피: 현미","tags":["ComplexCarb"]},
}

# ---------- 개인 규칙 ----------
def load_user_rules():
    defaults = {"avoid_keywords": ["현미","현미밥","brown rice"], "allow_keywords": ["커피"]}
    if os.path.exists(USER_RULES_PATH):
        try:
            with open(USER_RULES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
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
    # 없더라도 최소 스키마로 생성
    try:
        df = pd.read_csv(FOOD_DB_PATH, encoding="utf-8", engine="python")
    except Exception:
        df = pd.DataFrame(columns=["식품","식품군","등급","태그(영양)"])
        df.to_csv(FOOD_DB_PATH, index=False)
    if "태그(영양)" in df.columns:
        def parse_tags(x):
            try:
                return json.loads(x)
            except Exception:
                return [t.strip() for t in str(x).split(",") if t.strip()]
        df["태그(영양)"] = df["태그(영양)"].apply(parse_tags)
    else:
        df["태그(영양)"] = [[] for _ in range(len(df))]
    if "등급" not in df.columns:
        df["등급"] = "Safe"
    if "식품군" not in df.columns:
        df["식품군"] = ""
    return df

def save_food_db(df: pd.DataFrame):
    def to_jsonish(v):
        if isinstance(v, list): return json.dumps(v, ensure_ascii=False)
        try:
            parsed = json.loads(v)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            items = [t.strip() for t in str(v).split(",") if t.strip()]
            return json.dumps(items, ensure_ascii=False)
    if "태그(영양)" in df.columns:
        df["태그(영양)"] = df["태그(영양)"].apply(to_jsonish)
    df.to_csv(FOOD_DB_PATH, index=False, encoding="utf-8")

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

# --------- 자유입력 파싱 ---------
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
        if contains_any(name_norm, user_rules.get("avoid_keywords", [])):
            log = add_log_row(log, when_date, when_time, slot, "food", name_raw, qty, "", "Avoid", "개인 회피리스트", [""], source="memo(personal_avoid)")
            saved.append((name_raw, qty)); continue
        mapped, matched = match_food(name_norm, food_db)
        if matched:
            if mapped in VIRTUAL_RULES:
                vr = VIRTUAL_RULES[mapped]
                grade = vr["grade"]; flags = vr["flags"]; tags = vr["tags"]
                if contains_any(name_norm, user_rules.get("allow_keywords", [])):
                    grade = "Safe"; flags = "개인 허용"
                log = add_log_row(log, when_date, when_time, slot, "food", name_raw, qty, name_norm, grade, flags, tags, source="memo")
            else:
                rec = food_db[food_db["식품"]==mapped].iloc[0]
                grade = rec.get("등급","Safe")
                tags = rec.get("태그(영양)",[])
                if contains_any(name_norm, user_rules.get("allow_keywords", [])) and grade!="Avoid":
                    grade = "Safe"
                log = add_log_row(log, when_date, when_time, slot, "food", name_raw, qty, mapped, grade, "", tags, source="memo")
        else:
            grade, flags, tags = "", "", []
            if "빵" in name_norm: grade, flags = "Avoid", "글루텐/효모 가능성"; tags=["ComplexCarb"]
            if "햄" in name_norm: grade, flags = "Caution", "가공육/염분"; tags=["Protein"]
            if "치즈" in name_norm and ("비건" in name_norm or "베간" in name_norm): grade, flags = "Caution","가공 대체식"; tags=["HealthyFat"]
            if contains_any(name_norm, user_rules.get("allow_keywords", [])): grade="Safe"
            if contains_any(name_norm, user_rules.get("avoid_keywords", [])): grade="Avoid"; flags="개인 회피리스트"
            log = add_log_row(log, when_date, when_time, slot, "food", name_raw, qty, "", grade, flags, tags, source="memo(unmatched)")
        saved.append((name_raw, qty))
    return log, saved

# ---------- 점수 ----------
def score_day(df_log, df_food, date_str):
    day = df_log[(df_log["date"]==date_str) & (df_log["type"]=="food")]
    score = {k:0.0 for k in CORE_NUTRIENTS}
    for _, row in day.iterrows():
        fn = row.get("food_norm") or row.get("item")
        try: qty = float(row.get("qty") or 1.0)
        except Exception: qty = 1.0
        recs = df_food[df_food["식품"]==fn]
        if recs.empty:
            try: tags = json.loads(row.get("tags") or "[]")
            except Exception: tags = []
            for t in tags:
                if t in score: score[t] += qty
            continue
        tags = recs.iloc[0]["태그(영양)"]
        for t in tags:
            if t in score: score[t] += qty
    return score

# ---------- 현실형 PANTRY (fallback) ----------
# 흔히 구하는 식재료(유제품/글루텐/옥수수/콩류 과민 고려, 생선/육류/채소/과일/탄수/지방)
PANTRY = {
    "protein": ["대구","연어","닭가슴살","돼지고기","소고기","계란(알레르기 없을 때)"],
    "veg": ["양배추","당근","브로콜리","애호박","오이","시금치","상추","무"],
    "carb": ["쌀밥","고구마","감자","타피오카","퀴노아","옥수수죽(가능시)"],
    "fat": ["올리브유","들기름","아보카도(가능시)","참깨"],
    "fruit": ["사과","바나나","키위","블루베리","딸기"]
}

def build_baskets(df, include_caution=False):
    # 1) FoodDB에서 Safe/경계 추출
    pool = df.copy()
    pool = pool[pool["등급"].isin(["Safe","Caution"])] if include_caution else pool[pool["등급"]=="Safe"]
    def pick(col, cond):
        try:
            return pool[cond]["식품"].tolist()
        except Exception:
            return []
    proteins = pick("식품", (pool["식품군"].isin(["생선/해산물","육류"])) & (pool["태그(영양)"].apply(lambda t: "Protein" in t)))
    vegs = pick("식품", (pool["식품군"]=="채소") & (pool["태그(영양)"].apply(lambda t: "Fiber" in t)))
    carbs = pick("식품", (pool["태그(영양)"].apply(lambda t: "ComplexCarb" in t)))
    fats = pick("식품", (pool["태그(영양)"].apply(lambda t: "HealthyFat" in t)))
    fruits = pick("식품", (pool["식품군"]=="과일"))
    # 2) 부족하면 PANTRY로 보충
    def ensure(lst, pantry_list):
        s = list(dict.fromkeys(lst + [x for x in pantry_list if x not in lst]))
        return s
    proteins = ensure(proteins, PANTRY["protein"])
    vegs     = ensure(vegs, PANTRY["veg"])
    carbs    = ensure(carbs, PANTRY["carb"])
    fats     = ensure(fats, PANTRY["fat"])
    fruits   = ensure(fruits, PANTRY["fruit"])
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
        avoid_keywords += ["홍차","초콜릿","오렌지","레몬","라임","붉은 고추","스파이시","튀김","크림","토마토 소스"]
        if "커피" not in user_rules.get("allow_keywords", []):
            avoid_keywords += ["커피"]
        comp = {"protein":1,"veg":2,"carb":1,"fat":1}
        favor += ["LightProtein","Fiber"]
    elif mode=="더부룩":
        avoid_keywords += ["양파","마늘","강낭콩","렌틸","완두","콩","브로콜리","양배추","붉은 양배추","우유","요거트","치즈"]
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
    return avoid_keywords, comp, favor

def filter_keywords(items, kws):
    return [it for it in items if not any(k in it for k in kws)]

def filter_personal(items, user_rules):
    avoid = user_rules.get("avoid_keywords", [])
    return [it for it in items if not any(k in it for k in avoid)]

def pick_diverse(candidates, recent, need, rng):
    pool = [c for c in candidates if c not in recent]
    if len(pool) >= need:
        rng.shuffle(pool); return pool[:need]
    left = need - len(pool)
    rng.shuffle(pool)
    repeat_pool = [c for c in candidates if c in recent]
    rng.shuffle(repeat_pool)
    return pool + repeat_pool[:left]

def gen_meal(df_food, include_caution, mode, recent_items, favor_tags, rng, user_rules):
    baskets = build_baskets(df_food, include_caution=include_caution)
    avoid_kws, comp, favor_extra = mode_filters(mode, user_rules)
    for k in list(baskets.keys()):
        baskets[k] = filter_keywords(baskets[k], avoid_kws)
        baskets[k] = filter_personal(baskets[k], user_rules)
    local_favor = list(dict.fromkeys(list(favor_tags) + list(favor_extra)))
    def favor(lst):
        if not lst: return lst
        scored = []
        for name in lst:
            recs = df_food[df_food["식품"]==name]
            tags = recs.iloc[0]["태그(영양)"] if not recs.empty else []
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
    # 현실형 포맷: 타이틀 + 아이템
    title = build_meal_title(mode, meal)
    return title, meal

def build_meal_title(mode, items):
    # 간단 제목 생성: 주단백 + 보조
    if not items: return f"{mode} 제안"
    proteins = [x for x in items if any(k in x for k in ["대구","연어","닭","소고기","돼지고기","계란"])]
    main = proteins[0] if proteins else items[0]
    return f"{mode} • {main}"

def supplement_flag(text):
    if not text: return ("","")
    t = text.lower()
    for key, (grade,msg) in SUPP_ALERT_KEYWORDS.items():
        if key in t:
            return (grade, msg)
    return ("","")

# ---------- 앱 ----------
food_db = load_food_db()
log = ensure_log()
user_rules = load_user_rules()

st.title("🥣 민감도 식사 로그 • 현실형 제안")

# 사이드바: 개인 규칙
with st.sidebar:
    st.subheader("개인 규칙")
    avoid_str = st.text_input("회피 키워드(쉼표)", value=", ".join(user_rules.get("avoid_keywords", [])))
    allow_str = st.text_input("허용 키워드(쉼표)", value=", ".join(user_rules.get("allow_keywords", [])))
    if st.button("규칙 저장"):
        user_rules["avoid_keywords"] = [s.strip() for s in avoid_str.split(",") if s.strip()]
        user_rules["allow_keywords"] = [s.strip() for s in allow_str.split(",") if s.strip()]
        save_user_rules(user_rules)
        st.success("규칙 저장됨. 제안/파서 즉시 반영.")

tab1, tab2, tab3, tab4 = st.tabs(["📝 기록","📊 요약/제안","📤 내보내기","🛠 관리(편집/삭제)"])

with tab1:
    st.subheader("오늘 기록")
    d = st.date_input("날짜", value=date.today())
    slot = st.selectbox("슬롯(시간대)", SLOTS, index=2)
    t_input = st.time_input("시각", value=dtime(hour=12, minute=0))
    typ = st.radio("기록 종류", EVENT_TYPES, horizontal=True, index=0)
    if typ=="food":
        memo = st.text_area("메모 한 줄로 입력", height=100, placeholder="예: 쌀밥1, 대구구이1, 양배추찜1, 당근1, 올리브유0.5")
        if st.button("➕ 파싱해서 모두 저장", type="primary"):
            ds = d.strftime("%Y-%m-%d"); ts = t_input.strftime("%H:%M")
            log, saved = log_free_foods(log, ds, ts, slot, memo, food_db, user_rules)
            st.success(f"{len(saved)}개 항목 저장: " + ", ".join([f"{n}×{q}" for n,q in saved]))
    elif typ=="supplement":
        text = st.text_area("보충제/약/음료", height=80)
        g, flags = supplement_flag(text)
        if g=="Avoid": st.error(flags or "주의 보충제")
        elif g=="Caution": st.warning(flags or "경계 보충제")
        if st.button("➕ 저장", type="primary"):
            ds = d.strftime("%Y-%m-%d"); ts = t_input.strftime("%H:%M")
            log = add_log_row(log, ds, ts, slot, "supplement", text, 1.0, "", g, flags, [], source="manual")
            st.success("저장되었습니다.")
    else:
        text = st.text_area("증상(예: 속쓰림2, 더부룩1)", height=80)
        if st.button("➕ 저장", type="primary"):
            ds = d.strftime("%Y-%m-%d"); ts = t_input.strftime("%H:%M")
            log = add_log_row(log, ds, ts, slot, "symptom", text, 1.0, "", "", "", [], source="manual")
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

    # 부족 태그
    favor_tags = [n for n in ESSENTIALS if scores.get(n,0)<1]
    include_caution = st.checkbox("경계(Caution) 포함", value=False)
    diversity_n = st.slider("다양화(최근 N회 중복 회피)", min_value=0, max_value=10, value=5, step=1)
    # 최근 중복 회피
    recent_items = []
    if diversity_n>0:
        recent_df = log[log["type"]=="food"].sort_values(["date","time"]).tail(diversity_n*5)
        recent_items = (recent_df["food_norm"].fillna("") + "|" + recent_df["item"].fillna("")).tolist()
        recent_items = [x.split("|")[0] for x in recent_items if x]
    # 모드 선택
    mode = st.selectbox("제안 모드", SUGGEST_MODES, index=0)
    rng = random.Random(int(time.time()) % 10**9)

    cols = st.columns(3)
    for idx in range(3):
        title, meal = gen_meal(food_db, include_caution, mode, recent_items, favor_tags, rng, user_rules)
        with cols[idx]:
            st.markdown(f"**{title}**")
            if meal:
                st.write("• " + " / ".join(meal))
                if favor_tags: st.caption("부족 보완 우선 태그: " + ", ".join(favor_tags))
                # 저장 버튼
                if st.button(f"💾 이 조합 저장 (점심) — {idx+1}"):
                    now = datetime.now().strftime("%H:%M")
                    for token in meal:
                        # DB에 없더라도 그대로 기록 (food_norm은 그대로)
                        grade=""; tags=[]
                        rec = food_db[food_db["식품"]==token]
                        if not rec.empty:
                            grade = rec.iloc[0].get("등급","Safe"); tags = rec.iloc[0].get("태그(영양)",[])
                        add_log_row(log, date_str, now, "점심", "food", token, 1.0, token, grade, "", tags, source="suggested")
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
    # ZIP 백업
    if st.button("📦 전체 백업 ZIP 만들기"):
        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in [LOG_PATH, FOOD_DB_PATH, USER_RULES_PATH]:
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        zf.writestr(os.path.basename(p), f.read())
        mem_zip.seek(0)
        st.download_button("⬇️ 백업 ZIP 다운로드", data=mem_zip, file_name="meal_app_backup.zip", mime="application/zip")

with tab4:
    st.subheader("🛠 기록/DB 편집")
    # 로그 편집
    min_d = st.date_input("시작일", value=date.today())
    max_d = st.date_input("종료일", value=date.today())
    df = pd.read_csv(LOG_PATH)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        mask = (df["date"]>=min_d) & (df["date"]<=max_d)
        view = df[mask].copy().reset_index(drop=True)
        st.caption("셀 수정 후 '변경 저장'을 눌러 반영하세요.")
        edited = st.data_editor(view, num_rows="dynamic", use_container_width=True, key="edit_log")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("변경 저장"):
                df.loc[mask, :] = edited.values
                df["date"] = df["date"].astype(str)
                df.to_csv(LOG_PATH, index=False)
                st.success("로그 저장됨.")
        with c2:
            del_idx = st.multiselect("삭제할 행(뷰 인덱스)", options=list(range(len(view))))
            if st.button("선택 행 삭제"):
                to_drop = df[mask].iloc[del_idx].index
                df = df.drop(index=to_drop).reset_index(drop=True)
                df.to_csv(LOG_PATH, index=False)
                st.success(f"{len(del_idx)}개 행 삭제됨.")
    else:
        st.info("아직 로그가 없습니다.")

    st.markdown("---")
    # FoodDB 편집
    fdb = load_food_db()
    st.caption("태그(영양)은 JSON 배열 권장 예) [\"Protein\",\"Fiber\"].")
    fdb_edit = st.data_editor(fdb, num_rows="dynamic", use_container_width=True, key="edit_fooddb")
    if st.button("FoodDB 저장"):
        save_food_db(fdb_edit.copy())
        st.success("FoodDB 저장됨.")

    st.markdown("---")
    # user_rules 가져오기
    uploaded = st.file_uploader("user_rules.json 업로드(덮어쓰기)", type=["json"])
    if uploaded is not None:
        try:
            rules = json.load(uploaded)
            save_user_rules(rules)
            st.success("user_rules.json 업데이트 완료. 사이드바 확인.")
        except Exception as e:
            st.error(f"업로드 실패: {e}")
