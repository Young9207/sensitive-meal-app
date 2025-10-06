
import streamlit as st
import pandas as pd
import json, re, random, time, os, io, zipfile, math
from datetime import date, time as dtime, datetime

st.set_page_config(page_title="민감도 식사 로그 • 현실형 제안 (안정화)", page_icon="🥣", layout="wide")

def _force_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


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

# ---------- 유틸
def safe_json_loads(x):
    if isinstance(x, list): return x
    if x is None: return []
    if isinstance(x, (int, float)): return []
    s = str(x).strip()
    if s == "": return []
    # normalize single quotes to double
    if s.startswith("[") and "'" in s and '"' not in s:
        s = s.replace("'", '"')
    try:
        return json.loads(s)
    except Exception:
        # fallback: split by comma
        return [t.strip() for t in s.split(",") if t.strip()]

def safe_isnan(x):
    try:
        return math.isnan(x)
    except Exception:
        return False

# ---------- 개인 규칙
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

# ---------- I/O
def ensure_log():
    cols = ["date","weekday","time","slot","type","item","qty","food_norm","grade","flags","tags","source"]
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=cols).to_csv(LOG_PATH, index=False)
    try:
        log = pd.read_csv(LOG_PATH)
    except Exception:
        log = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in log.columns:
            log[c] = "" if c != "qty" else 0
    log = log[cols]
    # best-effort types
    log["qty"] = pd.to_numeric(log["qty"], errors="coerce").fillna(1.0)
    log.to_csv(LOG_PATH, index=False)
    return log

def load_food_db():
    base_cols = ["식품","식품군","등급","태그(영양)"]
    if not os.path.exists(FOOD_DB_PATH):
        pd.DataFrame(columns=base_cols).to_csv(FOOD_DB_PATH, index=False)
    try:
        df = pd.read_csv(FOOD_DB_PATH, encoding="utf-8", engine="python")
    except Exception:
        df = pd.DataFrame(columns=base_cols)
    for c in base_cols:
        if c not in df.columns: df[c] = "" if c!="태그(영양)" else "[]"
    # parse tags flexibly
    df["태그(영양)"] = df["태그(영양)"].apply(safe_json_loads)
    if "등급" not in df.columns: df["등급"] = "Safe"
    if "식품군" not in df.columns: df["식품군"] = ""
    return df[base_cols]

def save_food_db(df: pd.DataFrame):
    def to_jsonish(v):
        if isinstance(v, list): return json.dumps(v, ensure_ascii=False)
        return json.dumps(safe_json_loads(v), ensure_ascii=False)
    if "태그(영양)" in df.columns:
        df["태그(영양)"] = df["태그(영양)"].apply(to_jsonish)
    df.to_csv(FOOD_DB_PATH, index=False, encoding="utf-8")

def weekday_ko(dt: date):
    return ["MON","TUE","WED","THU","FRI","SAT","SUN"][dt.weekday()]

def add_log_row(log, date_str, t_str, slot, typ, item, qty, food_norm, grade, flags, tags, source="manual"):
    try:
        weekday = weekday_ko(datetime.strptime(date_str,"%Y-%m-%d").date())
    except Exception:
        weekday = ""
    new = pd.DataFrame([{
        "date": date_str,
        "weekday": weekday,
        "time": t_str,
        "slot": slot,
        "type": typ,
        "item": item,
        "qty": float(qty) if pd.notnull(qty) else 1.0,
        "food_norm": food_norm,
        "grade": grade,
        "flags": flags,
        "tags": json.dumps(tags, ensure_ascii=False) if isinstance(tags, list) else (tags or ""),
        "source": source
    }])
    # align columns
    cols = ["date","weekday","time","slot","type","item","qty","food_norm","grade","flags","tags","source"]
    for c in cols:
        if c not in log.columns: log[c] = "" if c != "qty" else 0
    new = new[cols]
    out = pd.concat([log[cols], new], ignore_index=True)
    out.to_csv(LOG_PATH, index=False)
    return out

# --------- 자유입력 파싱
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
    try:
        candidates = food_db[food_db["식품"].str.contains(name, case=False, na=False)]
        if not candidates.empty:
            return candidates.iloc[0]["식품"], True
    except Exception:
        pass
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

# ---------- 점수
def score_day(df_log, df_food, date_str):
    if df_log.empty: return {k:0.0 for k in CORE_NUTRIENTS}
    day = df_log[(df_log["date"]==date_str) & (df_log["type"]=="food")].copy()
    score = {k:0.0 for k in CORE_NUTRIENTS}
    for _, row in day.iterrows():
        fn = row.get("food_norm") or row.get("item")
        try: qty = float(row.get("qty") or 1.0)
        except Exception: qty = 1.0
        recs = df_food[df_food["식품"]==fn]
        if recs.empty:
            tags_val = row.get("tags")
            tags = safe_json_loads(tags_val)
            for t in tags:
                if t in score: score[t] += qty
            continue
        tags = recs.iloc[0]["태그(영양)"]
        for t in tags:
            if t in score: score[t] += qty
    return score

# ---------- 현실형 PANTRY

# --- Mode-specific anchor food pools (clearly different per condition) ---
MODE_ANCHORS = {
    "기본": {
        "protein": ["닭가슴살","연어","대구","돼지고기"],
        "veg": ["양배추","당근","브로콜리","애호박","오이","시금치"],
        "carb": ["쌀밥","고구마","감자","퀴노아","타피오카"],
        "fat": ["올리브유","들기름"],
        "fruit": ["사과","바나나","키위"]
    },
    "달다구리(당김)": {
        "protein": ["닭가슴살","대구"],
        "veg": ["오이","시금치","당근"],
        "carb": ["퀴노아","타피오카"],  # 급격혈당 피하기
        "fat": ["올리브유","아보카도(가능시)"],
        "fruit": ["블루베리","딸기","사과"]
    },
    "역류": {
        "protein": ["대구","닭가슴살"],
        "veg": ["오이","애호박","시금치","당근"],
        "carb": ["쌀죽","쌀밥","감자"],  # 부드러운 탄수화물
        "fat": ["올리브유"],
        "fruit": ["바나나","사과"]
    },
    "더부룩": {
        "protein": ["대구","닭가슴살"],
        "veg": ["오이","애호박","시금치","당근"],  # 저 FODMAP 위주
        "carb": ["쌀밥","감자","타피오카"],
        "fat": ["올리브유"],
        "fruit": ["바나나","키위"]
    },
    "붓기": {
        "protein": ["대구","닭가슴살","연어"],
        "veg": ["오이","시금치","당근"],
        "carb": ["고구마","감자","퀴노아"],
        "fat": ["올리브유"],
        "fruit": ["바나나","키위"]  # 칼륨
    },
    "피곤함": {
        "protein": ["소고기","돼지고기","연어"],  # 철/비타민B, 오메가3
        "veg": ["시금치","브로콜리","양배추"],
        "carb": ["고구마","퀴노아","쌀밥"],
        "fat": ["올리브유","들기름"],
        "fruit": ["키위","오렌지(허용시)","사과"]
    },
    "변비": {
        "protein": ["연어","닭가슴살"],
        "veg": ["양배추","브로콜리","시금치","당근"],
        "carb": ["퀴노아","고구마","쌀밥"],
        "fat": ["올리브유","들기름","참깨"],
        "fruit": ["키위","사과","바나나"]
    }
}


# ---- Availability filters ----
RARE_BLACKLIST = {
    # 너무 구하기 힘들거나 비현실적인 것들
    "따개비","멧돼지","타조","말고기","사슴고기","황새치","고둥","캐비어",
    "퍼츠","각시서대속 어류","참돔","먹도미류","바틀피시","해덕","농어",
}

COMMON_WHITELIST = {
    "protein": {"닭가슴살","대구","연어","돼지고기","소고기","계란(알레르기 없을 때)","고등어","참치(캔)"},
    "veg": {"양배추","당근","브로콜리","애호박","오이","시금치","상추","무","감자","고구마","파프리카","토마토"},
    "carb": {"쌀밥","쌀죽","고구마","감자","퀴노아","타피오카","옥수수죽(가능시)"},
    "fat": {"올리브유","들기름","참기름","아보카도(가능시)","참깨"},
    "fruit": {"사과","바나나","키위","블루베리","딸기","배"}
}

def apply_availability_filter(items, role_key, allow_rare=False):
    if allow_rare:
        # 그래도 희귀 블랙리스트는 제외
        return [x for x in items if x not in RARE_BLACKLIST]
    common = COMMON_WHITELIST.get(role_key, set())
    return [x for x in items if (x in common) and (x not in RARE_BLACKLIST)]

PANTRY = {
    "protein": ["대구","연어","닭가슴살","돼지고기","소고기","계란(알레르기 없을 때)"],
    "veg": ["양배추","당근","브로콜리","애호박","오이","시금치","상추","무"],
    "carb": ["쌀밥","고구마","감자","타피오카","퀴노아","옥수수죽(가능시)"],
    "fat": ["올리브유","들기름","아보카도(가능시)","참깨"],
    "fruit": ["사과","바나나","키위","블루베리","딸기"]
}

def build_baskets(df, include_caution=False):
    pool = df.copy()
    pool = pool[pool["등급"].isin(["Safe","Caution"])] if include_caution else pool[pool["등급"]=="Safe"]
    def pick(cond):
        try: return pool[cond]["식품"].tolist()
        except Exception: return []
    proteins = pick((pool["식품군"].isin(["생선/해산물","육류"])) & (pool["태그(영양)"].apply(lambda t: "Protein" in t)))
    vegs = pick((pool["식품군"]=="채소") & (pool["태그(영양)"].apply(lambda t: "Fiber" in t)))
    carbs = pick((pool["태그(영양)"].apply(lambda t: "ComplexCarb" in t)))
    fats = pick((pool["태그(영양)"].apply(lambda t: "HealthyFat" in t)))
    fruits = pick((pool["식품군"]=="과일"))
    def ensure(lst, pantry_list):
        return list(dict.fromkeys(lst + [x for x in pantry_list if x not in lst]))
    return {
        "protein": ensure(proteins, PANTRY["protein"]),
        "veg": ensure(vegs, PANTRY["veg"]),
        "carb": ensure(carbs, PANTRY["carb"]),
        "fat": ensure(fats, PANTRY["fat"]),
        "fruit": ensure(fruits, PANTRY["fruit"]),
    }

def mode_filters(mode, user_rules):
    """Return (avoid_keywords, composition, favor_tags, human_avoid_note)"""
    avoid_keywords = []
    comp = {"protein":1,"veg":2,"carb":1,"fat":1,"fruit":0}
    favor = []
    note = []
    if mode=="기본":
        pass
    elif mode=="달다구리(당김)":
        note += ["정제당/디저트류 제외"]
        comp = {"protein":1,"veg":1,"carb":0,"fat":1,"fruit":1}
        avoid_keywords += ["초콜릿","케이크","크림","튀김"]
        favor += ["C","Fiber","K_potassium","HealthyFat"]
    elif mode=="역류":
        note += ["시트러스/매운류/기름진 조리법 제외"]
        avoid_keywords += ["홍차","초콜릿","오렌지","레몬","라임","붉은 고추","스파이시","튀김","크림","토마토 소스"]
        if "커피" not in user_rules.get("allow_keywords", []):
            avoid_keywords += ["커피"]
        comp = {"protein":1,"veg":2,"carb":1,"fat":1}
        favor += ["LightProtein","Fiber"]
    elif mode=="더부룩":
        note += ["고FODMAP 재료(양파/마늘/콩류/양배추/브로콜리) 제외"]
        avoid_keywords += ["양파","마늘","강낭콩","렌틸","완두","콩","브로콜리","양배추","붉은 양배추","우유","요거트","치즈"]
        comp = {"protein":1,"veg":2,"carb":1,"fat":1}
        favor += ["LightProtein","Fiber"]
    elif mode=="붓기":
        note += ["염분/절임/가공육/간장 베이스 제외"]
        avoid_keywords += ["절임","젓갈","우메보시","김치","햄","베이컨","가공","스톡","간장"]
        comp = {"protein":1,"veg":2,"carb":1,"fat":1,"fruit":0}
        favor += ["K_potassium","Fiber","Hydration"]
    elif mode=="피곤함":
        note += ["튀김/크림/과음 제외"]
        avoid_keywords += ["튀김","크림","과음"]
        comp = {"protein":1,"veg":2,"carb":1,"fat":1}
        favor += ["B","Fe","Mg","ComplexCarb"]
    elif mode=="변비":
        note += ["유제품/튀김/저수분 식단 제외"]
        comp = {"protein":1,"veg":2,"carb":1,"fat":1,"fruit":1}
        avoid_keywords += ["치즈","크림","튀김"]
        favor += ["Fiber","Hydration","K_potassium","HealthyFat"]
    return avoid_keywords, comp, favor, ", ".join(note)

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

def gen_meal(df_food, include_caution, mode, recent_items, favor_tags, rng, user_rules, allow_rare=False):
    """Return (title, items, explain)
    - items are built from MODE_ANCHORS[mode] prioritized, then general baskets
    - explain shows what was avoided (mode note and personal rules)
    """
    baskets = build_baskets(df_food, include_caution=include_caution)
    avoid_kws, comp, favor_extra, mode_note = mode_filters(mode, user_rules)
    anchors = MODE_ANCHORS.get(mode, {})
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
        scored.sort(key=lambda x: (-x[0], random.random()))
        return [n for _, n in scored]
    for key in baskets.keys():
        # 1) 우선 앵커 병합
        if key in anchors:
            front = [x for x in anchors[key] if x in baskets[key]]
            rest = [x for x in baskets[key] if x not in front]
            merged = favor(front) + favor(rest)
        else:
            merged = favor(baskets[key])
        # 2) 가용성 필터 적용
        baskets[key] = apply_availability_filter(merged, key, allow_rare=allow_rare)
    meal = []
    for key, need in comp.items():
        chosen = pick_diverse(baskets[key], recent_items, need, rng)
        meal += chosen
    title = build_meal_title(mode, meal)
    explain = mode_note
    # Personal avoids surfaced
    pa = user_rules.get("avoid_keywords", [])
    if pa:
        explain = (explain + " | 개인 회피: " + ", ".join(pa)).strip(' |')
    return title, meal, explain

def build_meal_title(mode, items):
    if not items: return f"{mode} 제안"
    proteins = [x for x in items if any(k in x for k in ["대구","연어","닭","소고기","돼지고기","계란"])]
    main = proteins[0] if proteins else items[0]
    return f"{mode} • {main}"

def supplement_flag(text):
    if not text: return ("","")
    t = text.lower()
    for key, (grade,msg) in SUPP_ALERT_KEYWORDS.items():
        if key in t: return (grade, msg)
    return ("","")

# ---------- 앱 ----------
food_db = load_food_db()
log = ensure_log()
user_rules = load_user_rules()

st.title("🥣 민감도 식사 로그 • 현실형 제안 (안정화)")

with st.sidebar:
    st.subheader("개인 규칙")
    allow_rare = st.checkbox("희귀 식재료 포함", value=False, help="체크 해제 시 구하기 쉬운 재료만 제안합니다.")
    avoid_str = st.text_input("회피 키워드(쉼표)", value=", ".join(user_rules.get("avoid_keywords", [])))
    allow_str = st.text_input("허용 키워드(쉼표)", value=", ".join(user_rules.get("allow_keywords", [])))
    debug = st.checkbox("디버그 정보 표시", value=False)
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
            try:
                ds = d.strftime("%Y-%m-%d"); ts = t_input.strftime("%H:%M")
                log, saved = log_free_foods(log, ds, ts, slot, memo, food_db, user_rules)
                st.success(f"{len(saved)}개 항목 저장: " + ", ".join([f"{n}×{q}" for n,q in saved])); _force_rerun()
            except Exception as e:
                st.error("파싱/저장 중 오류가 발생했습니다.")
                if debug: st.exception(e)
    elif typ=="supplement":
        text = st.text_area("보충제/약/음료", height=80)
        g, flags = supplement_flag(text)
        if g=="Avoid": st.error(flags or "주의 보충제")
        elif g=="Caution": st.warning(flags or "경계 보충제")
        if st.button("➕ 저장", type="primary"):
            try:
                ds = d.strftime("%Y-%m-%d"); ts = t_input.strftime("%H:%M")
                log = add_log_row(log, ds, ts, slot, "supplement", text, 1.0, "", g, flags, [], source="manual")
                st.success("저장되었습니다."); _force_rerun()
            except Exception as e:
                st.error("저장 중 오류가 발생했습니다.")
                if debug: st.exception(e)
    else:
        text = st.text_area("증상(예: 속쓰림2, 더부룩1)", height=80)
        if st.button("➕ 저장", type="primary"):
            try:
                ds = d.strftime("%Y-%m-%d"); ts = t_input.strftime("%H:%M")
                log = add_log_row(log, ds, ts, slot, "symptom", text, 1.0, "", "", "", [], source="manual")
                st.success("저장되었습니다."); _force_rerun()
            except Exception as e:
                st.error("저장 중 오류가 발생했습니다.")
                if debug: st.exception(e)
    st.markdown("---")
    st.caption("최근 기록")
    try:
        fresh = ensure_log()
        tmp = fresh.copy()
        tmp["date"] = tmp["date"].astype(str)
        tmp["time"] = tmp["time"].astype(str)
        st.dataframe(tmp.sort_values(['date','time']).tail(20), use_container_width=True, height=240)
    except Exception as e:
        st.error("최근 기록 표시 중 오류")
        if debug: st.exception(e)

with tab2:
    st.subheader("요약 & 다음 끼니 제안(3가지)")
    dsum = st.date_input("기준 날짜", value=date.today(), key="sumdate_2")
    date_str = dsum.strftime("%Y-%m-%d")
    try:
        scores = score_day(log, food_db, date_str)
        score_df = pd.DataFrame([scores]).T.reset_index()
        score_df.columns = ["영양소","점수"]
        st.dataframe(score_df, use_container_width=True, height=260)
    except Exception as e:
        st.error("점수 계산 중 오류")
        if debug: st.exception(e)

    favor_tags = [n for n in ESSENTIALS if scores.get(n,0)<1] if 'scores' in locals() else []
    include_caution = st.checkbox("경계(Caution) 포함", value=False)
    diversity_n = st.slider("다양화(최근 N회 중복 회피)", min_value=0, max_value=10, value=5, step=1)
    recent_items = []
    try:
        if diversity_n>0:
            r = ensure_log()
            r = r[r["type"]=="food"].copy()
            r["date"] = r["date"].astype(str)
            r["time"] = r["time"].astype(str)
            recent_df = r.sort_values(["date","time"]).tail(diversity_n*5)
            recent_items = (recent_df["food_norm"].fillna("") + "|" + recent_df["item"].fillna("")).tolist()
            recent_items = [x.split("|")[0] for x in recent_items if x]
    except Exception as e:
        if debug: st.exception(e)

    mode = st.selectbox("제안 모드", SUGGEST_MODES, index=0)
    if "suggest_seed" not in st.session_state:
        st.session_state.suggest_seed = 0
    if st.button("🔄 새로고침"):
        st.session_state.suggest_seed += 1
    rng = random.Random(hash((mode, st.session_state.suggest_seed)) % (10**9))

    cols = st.columns(3)
    for idx in range(3):
        try:
            title, meal, explain = gen_meal(
                food_db,
                include_caution,
                mode,
                recent_items,
                favor_tags,
                rng,
                user_rules,
                allow_rare=allow_rare
            )
            with cols[idx]:
                st.markdown(f"**{title}**")
                if meal:
                    st.write("• " + " / ".join(meal))
                    if favor_tags: st.caption("부족 보완 우선 태그: " + ", ".join(favor_tags))
                    if explain:
                        st.caption("모드 적용: " + explain)
                    if st.button(f"💾 이 조합 저장 (점심) — {idx+1}"):
                        now = datetime.now().strftime("%H:%M")
                        for token in meal:
                            grade=""; tags=[]
                            rec = food_db[food_db["식품"]==token]
                            if not rec.empty:
                                grade = rec.iloc[0].get("등급","Safe"); tags = rec.iloc[0].get("태그(영양)",[])
                            add_log_row(log, date_str, now, "점심", "food", token, 1.0, token, grade, "", tags, source="suggested")
                        st.success("저장 완료! 기록 탭에서 확인하세요.")
                else:
                    st.info("조건을 만족하는 식품 풀이 부족합니다. 개인 회피리스트 또는 FoodDB를 확인하세요.")
        except Exception as e:
            with cols[idx]:
                st.error("제안 생성 중 오류")
                if debug: st.exception(e)
                
with tab3:
    st.subheader("내보내기/백업")
    try:
        # 개별 파일 다운로드 (존재할 때만 노출)
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "rb") as f:
                st.download_button(
                    "⬇️ log.csv 다운로드",
                    data=f.read(),
                    file_name="log.csv",
                    mime="text/csv"
                )
        if os.path.exists(FOOD_DB_PATH):
            with open(FOOD_DB_PATH, "rb") as f:
                st.download_button(
                    "⬇️ food_db.csv 다운로드",
                    data=f.read(),
                    file_name="food_db.csv",
                    mime="text/csv"
                )
        if os.path.exists(USER_RULES_PATH):
            with open(USER_RULES_PATH, "rb") as f:
                st.download_button(
                    "⬇️ user_rules.json 다운로드",
                    data=f.read(),
                    file_name="user_rules.json",
                    mime="application/json"
                )

        # ZIP 백업 만들기
        if st.button("📦 전체 백업 ZIP 만들기"):
            mem_zip = io.BytesIO()
            with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in [LOG_PATH, FOOD_DB_PATH, USER_RULES_PATH]:
                    if p and os.path.exists(p):
                        # 파일을 메모리에 읽지 않고 바로 추가
                        zf.write(p, arcname=os.path.basename(p))
            mem_zip.seek(0)
            st.download_button(
                "⬇️ 백업 ZIP 다운로드",
                data=mem_zip,
                file_name="meal_app_backup.zip",
                mime="application/zip"
            )
    except Exception as e:
        st.error("내보내기 중 오류가 발생했습니다.")
        if debug:
            st.exception(e)

with tab4:
    st.subheader("🛠 기록/DB 편집 & 복구")
    # --- 로그 편집 ---
    min_d = st.date_input("시작일", value=date.today())
    max_d = st.date_input("종료일", value=date.today())
    try:
        df = pd.read_csv(LOG_PATH)
    except Exception:
        df = pd.DataFrame(columns=["date","weekday","time","slot","type","item","qty","food_norm","grade","flags","tags","source"])
        df.to_csv(LOG_PATH, index=False)
    if not df.empty:
        try:
            # normalize types for view only
            dfv = df.copy()
            dfv["date"] = pd.to_datetime(dfv["date"], errors="coerce").dt.date
            mask = (dfv["date"]>=min_d) & (dfv["date"]<=max_d)
            view = dfv[mask].copy()
            view = view.reset_index()  # keep original index
            st.caption("셀 수정 후 '변경 저장'을 눌러 반영하세요. 행 추가는 아래 규칙으로 저장되며, 삭제는 오른쪽 기능 사용.")
            edited = st.data_editor(view.drop(columns=["index"]), num_rows="dynamic", use_container_width=True, key="edit_log")

            c1,c2,c3 = st.columns(3)
            with c1:
                if st.button("변경 저장"):
                    try:
                        # 1) update existing rows (by original index)
                        min_len = min(len(edited), len(view))
                        for col in edited.columns:
                            df.loc[view.iloc[:min_len]["index"], col] = edited.iloc[:min_len][col].values
                        # 2) append new rows if any
                        if len(edited) > len(view):
                            extra = edited.iloc[len(view):].copy()
                            # ensure required columns exist
                            for c in df.columns:
                                if c not in extra.columns:
                                    extra[c] = "" if c != "qty" else 1.0
                            # convert date to str
                            if "date" in extra.columns:
                                extra["date"] = extra["date"].astype(str)
                            df = pd.concat([df, extra[df.columns]], ignore_index=True)
                        # 3) finalize types & save
                        df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(1.0)
                        df["date"] = df["date"].astype(str)
                        df.to_csv(LOG_PATH, index=False)
                        st.success("로그 저장됨."); _force_rerun()
                    except Exception as e:
                        st.error("로그 저장 중 오류")
                        if debug: st.exception(e)
            with c2:
                del_idx = st.multiselect("삭제할 행(뷰 인덱스)", options=list(range(len(view))))
                if st.button("선택 행 삭제"):
                    try:
                        to_drop = view.iloc[del_idx]["index"].tolist()
                        df = df.drop(index=to_drop).reset_index(drop=True)
                        df.to_csv(LOG_PATH, index=False)
                        st.success(f"{len(del_idx)}개 행 삭제됨."); _force_rerun()
                    except Exception as e:
                        st.error("행 삭제 중 오류")
                        if debug: st.exception(e)
            with c3:
                if st.button("파일 복구(깨졌을 때 초기화)"):
                    try:
                        backup_name = f"log_backup_{int(time.time())}.csv"
                        if os.path.exists(LOG_PATH):
                            os.replace(LOG_PATH, backup_name)
                        pd.DataFrame(columns=["date","weekday","time","slot","type","item","qty","food_norm","grade","flags","tags","source"]).to_csv(LOG_PATH, index=False)
                        st.success(f"복구 완료. 기존 파일은 {backup_name} 로 백업됨."); _force_rerun()
                    except Exception as e:
                        st.error("복구 실패")
                        if debug: st.exception(e)
        except Exception as e:
            st.error("로그 편집 UI 구성 중 오류")
            if debug: st.exception(e)
    else:
        st.info("아직 로그가 없습니다.")

    st.markdown("---")
    # --- user_rules 가져오기 ---
    uploaded = st.file_uploader("user_rules.json 업로드(덮어쓰기)", type=["json"])
    if uploaded is not None:
        try:
            rules = json.load(uploaded)
            save_user_rules(rules)
            st.success("user_rules.json 업데이트 완료. 사이드바 확인."); _force_rerun()
        except Exception as e:
            st.error(f"업로드 실패: {e}")
