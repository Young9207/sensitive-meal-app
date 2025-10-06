
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



# === nutrient_dict.csv 연동 ===
NUTRIENT_TIPS = dict(globals().get("NUTRIENT_TIPS", {}))  # 짧은 한줄설명
NUTRIENT_TIPS_LONG = dict(globals().get("NUTRIENT_TIPS_LONG", {}))  # 자세한설명
BENEFIT_MAP = dict(globals().get("BENEFIT_MAP", {}))  # 태그 → 베네핏 라벨/설명
NUTRIENT_EXAMPLES = dict(globals().get("NUTRIENT_EXAMPLES", {}))  # 영양소 → 대표식품 예시 리스트

def _load_nutrient_dict_csv(paths=("data/nutrient_dict.csv", "/mnt/data/nutrient_dict.csv")):
    """
    nutrient_dict.csv 스키마:
      - 영양소
      - 한줄설명
      - 자세한설명
      - 혜택라벨(요약)
      - 대표식품(쉼표로구분)
    """
    import pandas as _pd, os as _os
    for _p in paths:
        try:
            if _os.path.exists(_p):
                _df = _pd.read_csv(_p)
                for _, _r in _df.iterrows():
                    key = str(_r.get("영양소") or "").strip()
                    if not key:
                        continue
                    short = str(_r.get("한줄설명") or "").strip()
                    long = str(_r.get("자세한설명") or "").strip()
                    label = str(_r.get("혜택라벨(요약)") or "").strip()
                    examples = str(_r.get("대표식품(쉼표로구분)") or "").strip()

                    if short:
                        NUTRIENT_TIPS[key] = short
                    if long:
                        NUTRIENT_TIPS_LONG[key] = long
                    # BENEFIT_MAP은 가능한 간결한 라벨을 우선
                    if label:
                        BENEFIT_MAP[key] = label
                    elif short:
                        BENEFIT_MAP[key] = short

                    if examples:
                        NUTRIENT_EXAMPLES[key] = [x.strip() for x in examples.split(",") if x.strip()]
                return True
        except Exception:
            pass
    return False

# 최초 로드 시도
try:
    _nd_ok = _load_nutrient_dict_csv()
    if _nd_ok:
        if debug:
            st.caption("✅ nutrient_dict.csv 로드됨")
    else:
        if debug:
            st.caption("ℹ️ nutrient_dict.csv를 찾지 못했거나 컬럼이 맞지 않습니다.")
except Exception as _e:
    if 'st' in globals() and debug:
        st.warning("nutrient_dict.csv 로드 중 오류 발생")
        st.exception(_e)
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

    # === 영양 설명/보강 제안 ===
    with st.expander("영양 설명과 보강 아이디어 보기", expanded=False):
        try:
            low_keys = [k for k, v in sorted(scores.items(), key=lambda x: x[1]) if v < 1.0]
            if not low_keys:
                st.markdown("- 오늘은 핵심 영양소 커버가 전반적으로 **양호**합니다.")
            else:
                st.markdown("부족/미달 영양소와 간단 설명:")
                rows = []
                for k in low_keys:
                    tip = _lookup_tip(k)
                    rows.append(f"- **{_friendly_label(k)}**: {tip}")
                st.markdown("\\n".join(rows))

                # 예시 식품 추천 (food_db의 태그 기반)
                try:
                    eg_lines = []
                    # 태그에 k 또는 해당 한글명이 포함된 식품 예시 추출
                    for k in low_keys[:5]:
                        tag_candidates = {k, _nut_ko(k), _nut_en(k)}
                        cand = []
                        for _, r in food_db.iterrows():
                            tags = r.get("태그(영양)", [])
                            if not isinstance(tags, list):
                                continue
                            tset = set(map(str, tags))
                            if tset & tag_candidates:
                                cand.append(str(r.get("식품")))
                            if len(cand) >= 6:
                                break
                        if cand:
                            eg_lines.append(f"  • **{_friendly_label(k)}** 예시: " + ", ".join(sorted(set(cand))[:6]))
                    if eg_lines:
                        st.markdown("보강에 도움이 되는 식품 예시:")
                        st.markdown("\\n".join(eg_lines))
                except Exception as _e:
                    pass
        except Exception as e:
            st.info("설명 생성 중 문제가 있었습니다.")
            if debug: st.exception(e)

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

# ==== [ADDON] 즉석 식단 평가 + 영양 한줄 코멘트 ===============================
try:
    import streamlit as st
    import pandas as pd
    import re
    import random
    from difflib import get_close_matches
except Exception:
    pass

if 'CORE_NUTRIENTS' not in globals():
    CORE_NUTRIENTS = [
        "단백질", "식이섬유", "철", "칼슘", "마그네슘", "칼륨",
        "오메가3", "비타민A", "비타민B", "비타민C", "비타민D", "비타민E",
        "저당", "저염", "건강한지방"
    ]

if 'ESSENTIALS' not in globals():
    ESSENTIALS = ["단백질", "식이섬유", "비타민C", "칼슘"]

if 'food_db' not in globals():
    FOOD_ROWS = [
        ("닭가슴살", "Safe", ["단백질", "저지방"]),
        ("두부", "Safe", ["단백질", "칼슘"]),
        ("연어", "Safe", ["단백질", "오메가3", "건강한지방"]),
        ("계란", "Safe", ["단백질", "비타민D"]),
        ("대구구이", "Safe", ["단백질"]),
        ("현미밥", "Safe", ["식이섬유", "마그네슘"]),
        ("귀리", "Safe", ["식이섬유", "철"]),
        ("통밀빵", "Caution", ["식이섬유"]),
        ("쌀밥", "Safe", []),
        ("시금치", "Safe", ["철", "비타민A", "마그네슘"]),
        ("브로콜리", "Safe", ["비타민C", "식이섬유", "칼슘"]),
        ("양배추", "Safe", ["비타민C", "식이섬유"]),
        ("당근", "Safe", ["비타민A"]),
        ("버섯", "Safe", ["비타민B", "식이섬유"]),
        ("올리브유", "Safe", ["건강한지방", "비타민E"]),
        ("아보카도", "Safe", ["건강한지방", "칼륨", "식이섬유"]),
        ("아몬드", "Caution", ["건강한지방", "비타민E", "칼슘"]),
        ("요거트", "Caution", ["칼슘", "단백질"]),
    ]
    food_db = pd.DataFrame(FOOD_ROWS, columns=["식품", "등급", "태그(영양)"])

if 'NUTRIENT_TIPS' not in globals():
    NUTRIENT_TIPS = {
        "단백질": "근육 유지·포만감 도움—식사 후 허기 감소.",
        "식이섬유": "배변 리듬·포만감, 당 흡수 완만.",
        "철": "피로감 감소·집중 도움(산소 운반).",
        "칼슘": "뼈·치아 기본, 근육 수축에도 필요.",
        "마그네슘": "긴장 완화·수면·근육 기능 도움.",
        "칼륨": "나트륨 배출로 붓기·혈압 관리에 유리.",
        "오메가3": "심혈관·염증 균형 도움(등푸른 생선).",
        "비타민A": "눈·피부 점막 보호(색 진한 채소).",
        "비타민B": "에너지 대사 서포트, 피로 완화.",
        "비타민C": "면역·철 흡수 UP(가열 덜 한 채소/과일).",
        "비타민D": "뼈 건강·면역 도움(햇빛·계란·생선).",
        "비타민E": "항산화로 세포 보호·피부 컨디션.",
        "저당": "식후 혈당 출렁임 완화.",
        "저염": "붓기·혈압 관리에 도움.",
        "건강한지방": "포만감·지용성 비타민 흡수에 도움.",
        "저지방": "열량 대비 단백질 확보에 유리."
    }

if 'NUTRIENT_TIPS_LONG' not in globals():
    NUTRIENT_TIPS_LONG = {
        "단백질": "근육 유지, 상처 회복, 포만감 유지에 핵심.",
        "식이섬유": "배변 규칙성, 포만감, 혈당 급상승 완화에 도움.",
        "철": "피로감·어지러움 예방(산소 운반). 비타민 C와 함께 섭취하면 흡수↑",
        "칼슘": "뼈·치아 건강, 신경·근육 기능.",
        "마그네슘": "근육 이완, 수면·긴장 완화, 에너지 대사.",
        "칼륨": "나트륨 배출을 도와 붓기·혈압 조절.",
        "오메가3": "심혈관·뇌 건강, 염증 균형.",
        "비타민A": "야간 시력·피부·점막 보호.",
        "비타민B": "에너지 생성·피로 완화(복합군).",
        "비타민C": "면역, 철 흡수, 항산화.",
        "비타민D": "칼슘 흡수·뼈 건강, 면역 조절.",
        "비타민E": "항산화(세포 보호), 피부 컨디션.",
        "저당": "식후 혈당 출렁임 감소.",
        "저염": "붓기 완화·혈압 관리.",
        "건강한지방": "포만감·지용성 비타민 흡수 도우미."
    }

if 'NUTRIENT_SOURCES' not in globals():
    NUTRIENT_SOURCES = {
        "단백질": ["닭가슴살", "두부", "연어", "계란", "대구구이", "요거트"],
        "식이섬유": ["현미밥", "귀리", "브로콜리", "양배추", "아보카도", "버섯"],
        "철": ["시금치", "귀리", "붉은살 생선", "콩류"],
        "칼슘": ["두부", "브로콜리", "요거트", "아몬드"],
        "마그네슘": ["현미밥", "시금치", "견과류"],
        "칼륨": ["아보카도", "바나나", "감자", "시금치"],
        "오메가3": ["연어", "등푸른 생선", "호두"],
        "비타민A": ["당근", "시금치", "호박"],
        "비타민B": ["버섯", "통곡물", "달걀"],
        "비타민C": ["브로콜리", "양배추", "키위", "파프리카"],
        "비타민D": ["계란", "연어", "버섯(일광 건조)"],
        "비타민E": ["올리브유", "아몬드", "아보카도"],
        "저당": ["채소 위주 반찬", "통곡물 소량", "무가당 요거트"],
        "저염": ["구운/찐 조리", "양념절제", "허브·레몬 활용"],
        "건강한지방": ["올리브유", "아보카도", "견과류"]
    }

if 'BENEFIT_MAP' not in globals():
    BENEFIT_MAP = {
        "단백질": "근육·포만감",
        "식이섬유": "장건강·포만감·혈당완화",
        "칼슘": "뼈·치아",
        "비타민D": "뼈·면역",
        "비타민C": "면역·철흡수",
        "오메가3": "심혈관·염증완화",
        "칼륨": "붓기·혈압",
        "마그네슘": "긴장완화·수면",
        "비타민E": "항산화·피부",
        "비타민A": "눈·피부",
        "비타민B": "에너지대사"
    }

if 'VIRTUAL_RULES' not in globals():
    VIRTUAL_RULES = {}

def _split_free_text(text: str):
    if not text:
        return []
    return [p.strip() for p in re.split(r"[,|\n]+", text) if p.strip()]

def _parse_qty(token: str):
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*$", token)
    if m:
        qty = float(m.group(1))
        name = token[:m.start()].strip()
        return name, qty
    return token.strip(), 1.0

def _contains_any(text: str, keywords):
    t = (text or "").lower()
    for k in (keywords or []):
        if k.lower() in t:
            return True
    return False

def _match_food(name: str, df):
    names = df["식품"].tolist()
    if name in names:
        return name, True
    cand = get_close_matches(name, names, n=1, cutoff=0.6)
    if cand:
        return cand[0], True
    base = re.sub(r"(구이|볶음|찜|샐러드|수프|조림|구운|생)", "", name).strip()
    if base and base != name:
        cand = get_close_matches(base, names, n=1, cutoff=0.6)
        if cand:
            return cand[0], True
    return name, False

def _score_tokens(free_text, df_food, user_rules):
    tokens = _split_free_text(free_text)
    rows = []
    score = {k: 0.0 for k in CORE_NUTRIENTS}
    for tok in tokens:
        name_raw, qty = _parse_qty(tok)
        name_norm = (name_raw or "").strip()
        if not name_norm:
            continue
        if _contains_any(name_norm, user_rules.get("avoid_keywords", [])):
            rows.append({"식품": name_raw, "정규화": name_norm, "수량": qty,
                         "등급": "Avoid", "사유": "개인 회피리스트", "태그(영양)": []})
            continue
        mapped, matched = _match_food(name_norm, df_food)
        tags, grade, flags = [], "Safe", ""
        if matched:
            if mapped in VIRTUAL_RULES:
                vr = VIRTUAL_RULES[mapped]
                grade, flags, tags = vr.get("grade", "Safe"), vr.get("flags", ""), vr.get("tags", [])
                if _contains_any(name_norm, user_rules.get("allow_keywords", [])):
                    grade, flags = "Safe", "개인 허용"
            else:
                rec = df_food[df_food["식품"] == mapped].iloc[0]
                grade = rec.get("등급", "Safe")
                tags = rec.get("태그(영양)", [])
                if _contains_any(name_norm, user_rules.get("allow_keywords", [])) and grade != "Avoid":
                    grade = "Safe"
        else:
            grade, flags, tags = "Unknown", "DB 미등재", []
        for t in tags:
            if t in score:
                score[t] += float(qty or 1.0)
        rows.append({
            "식품": name_raw,
            "정규화": mapped if matched else name_norm,
            "수량": qty,
            "등급": grade,
            "사유": flags,
            "태그(영양)": tags
        })
    return score, pd.DataFrame(rows)

def _ensure_log():
    try:
        return ensure_log()
    except Exception:
        return pd.DataFrame(columns=["type", "date", "time", "food_norm", "item"])

def _tokens_from_today_log():
    import datetime as _dt
    df = _ensure_log()
    if df is None or df.empty:
        return []
    today = _dt.datetime.now().date()
    try:
        df['date'] = pd.to_datetime(df['date']).dt.date
    except Exception:
        return []
    df_today = df[(df['type'] == 'food') & (df['date'] == today)].copy()
    now_time = _dt.datetime.now().time()
    try:
        df_today['time'] = pd.to_datetime(df_today['time'].astype(str), errors='coerce').dt.time
        df_today = df_today[df_today['time'].isna() | (df_today['time'] <= now_time)]
    except Exception:
        pass
    tokens = []
    for _, r in df_today.iterrows():
        token = (str(r.get('item') or '')).strip() or (str(r.get('food_norm') or '')).strip()
        if token:
            tokens.append(token)
    return tokens

def _today_food_log_df():
    import datetime as _dt
    df = _ensure_log()
    if df is None or df.empty:
        return pd.DataFrame(columns=["type","date","time","food_norm","item","_dt","시간대"])
    try:
        df = df[df["type"] == "food"].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    except Exception:
        return pd.DataFrame(columns=["type","date","time","food_norm","item","_dt","시간대"])
    today = pd.Timestamp.now().normalize()
    df = df[df["date"].dt.normalize() == today]
    def _parse_dt(row):
        try:
            t = pd.to_datetime(str(row.get("time") or ""), errors="coerce").time()
        except Exception:
            t = None
        d = row["date"].date()
        if t is None:
            return pd.Timestamp.combine(d, pd.Timestamp.now().time())
        return pd.Timestamp.combine(d, t)
    if not df.empty:
        df["_dt"] = df.apply(_parse_dt, axis=1)
        def _tod_label(ts):
            h = ts.hour
            if 5 <= h < 11: return "아침"
            if 11 <= h < 16: return "점심"
            if 16 <= h < 21: return "저녁"
            return "간식"
        df["시간대"] = df["_dt"].apply(_tod_label)
        df = df.sort_values("_dt")
    else:
        df["_dt"] = pd.NaT
        df["시간대"] = ""
    return df

def _per_meal_breakdown(df_food, df_today):
    rows = []
    for _, r in df_today.iterrows():
        raw = str(r.get("item") or "").strip() or str(r.get("food_norm") or "").strip()
        if not raw:
            continue
        mapped, matched = _match_food(raw, df_food)
        tags, benefits = [], []
        if matched:
            try:
                rec = df_food[df_food["식품"] == mapped].iloc[0]
                tags = _parse_tags_flexible(rec.get("태그(영양)", []))
            except Exception:
                tags = []
        for t in tags:
            b = (_benefit_from_tag(t) or _lookup_tip(t))
            if b and b not in benefits:
                benefits.append(b)
        rows.append({
            "시간대": r.get("시간대", ""),
            "시각": r.get("_dt"),
            "입력항목": raw,
            "매칭식품": mapped if matched else raw,
            "채워진태그": ", ".join(tags[:5]),
            "직관설명": (" · ".join([x for x in benefits if x][:3]) or "균형 잡힌 선택")
        })
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(["시간대","시각"])
    return df_out

def _make_intuitive_summary(scores: dict, threshold: float = 1.0) -> str:
    filled_benefits = []
    low_benefits = []
    ordered_keys = list(ESSENTIALS) + [k for k in BENEFIT_MAP.keys() if k not in ESSENTIALS]
    for k in ordered_keys:
        val = float(scores.get(k, 0) or 0)
        benefit = BENEFIT_MAP.get(k)
        if not benefit:
            continue
        if val >= threshold:
            if benefit not in filled_benefits:
                filled_benefits.append(benefit)
        else:
            if benefit not in low_benefits:
                low_benefits.append(benefit)
    left = " · ".join(filled_benefits[:3]) if filled_benefits else ""
    right = " · ".join(low_benefits[:3]) if low_benefits else ""
    if left and right:
        return f"오늘 한 줄 요약: {left}는 꽤 채워졌고, {right}는 보충이 필요해요."
    elif left:
        return f"오늘 한 줄 요약: {left}는 잘 챙겨졌어요."
    elif right:
        return f"오늘 한 줄 요약: {right} 보충이 필요해요."
    else:
        return "오늘 한 줄 요약: 분석할 항목이 없어요."

try:
    st.divider()
    with st.container():
        st.header("⚡ 즉석 식단 평가 (저장 없이 분석)")

        with st.expander("📘 영양소 한눈 요약 (무엇에 좋은가 + 대표 식품)", expanded=False):
            df_gloss = pd.DataFrame([
                {
                    "영양소": k,
                    "무엇에 좋은가(쉽게)": NUTRIENT_TIPS_LONG.get(k, NUTRIENT_TIPS.get(k, "")),
                    "대표 식품": ", ".join(NUTRIENT_SOURCES.get(k, [])[:4])
                }
                for k in CORE_NUTRIENTS if (k in NUTRIENT_TIPS or k in NUTRIENT_TIPS_LONG)
            ])
            st.dataframe(df_gloss, use_container_width=True, height=380)
            st.caption("• 부족 태그가 뜨면 대표 식품을 참고해 다음 식사를 구성해보세요.")

        colA, colB, colC, colD = st.columns([1.2, 1.2, 1, 1])
        with colA:
            avoid = st.text_input("회피 키워드(쉼표)", value="")
        with colB:
            allow = st.text_input("허용 키워드(쉼표)", value="")
        with colC:
            include_caution = st.checkbox("Caution 포함", value=False)
        with colD:
            diversity_n = st.slider("다양화(최근 N회)", 0, 10, 5, 1)

        user_rules_local = {
            "avoid_keywords": [x.strip() for x in avoid.split(",") if x.strip()],
            "allow_keywords": [x.strip() for x in allow.split(",") if x.strip()],
        }

        source_mode = st.radio("분석 소스", ["오늘 기록 사용", "직접 입력"], horizontal=True, index=0)
        sample = "쌀밥1, 대구구이1, 양배추1, 당근1, 올리브유0.5"
        text_in = st.text_area(
            "식단 텍스트 (쉼표/줄바꿈 구분)",
            height=120,
            placeholder=sample,
            disabled=(source_mode == "오늘 기록 사용")
        )

        if source_mode == "오늘 기록 사용":
            _toks = _tokens_from_today_log()
            if _toks:
                st.caption("오늘 기록에서 불러온 항목: " + ", ".join(_toks))
                text_in = ", ".join(_toks)
            else:
                st.info("오늘 날짜의 음식 기록이 없어요. 직접 입력으로 전환해 주세요.")

        analyze = st.button("분석하기", type="primary")
        if analyze:
            try:
                scores, items_df = _score_tokens(text_in, food_db, user_rules_local)

                st.markdown("#### 🍱 파싱 결과")
                if items_df.empty:
                    st.info("항목이 없습니다. 식단을 입력해 주세요.")
                else:
                    st.dataframe(items_df, use_container_width=True, height=260)

                st.markdown("#### 🧭 태그 점수 + 한줄 설명")
                score_df = (
                    pd.DataFrame([scores]).T
                    .reset_index().rename(columns={"index": "영양소", 0: "점수"})
                    .sort_values("점수", ascending=False)
                )
                score_df["영양소(보기)"] = score_df["영양소"].map(_friendly_label)
                score_df["한줄설명"] = score_df["영양소"].map(lambda x: _lookup_tip(x))
                st.dataframe(score_df, use_container_width=True, height=320)

                missing = [n for n in ESSENTIALS if scores.get(n, 0) < 1]
                if missing:
                    tips_list = [f"- **{_friendly_label(n)}**: {(BENEFIT_MAP.get(_canon_key(n)) or NUTRIENT_TIPS.get(_canon_key(n), ''))}
   예시: {', '.join(_example_foods_for(n))}" for n in missing]
                    st.warning("부족 태그:\n" + "\n".join(tips_list))
                else:
                    st.success("핵심 태그 충족! (ESSENTIALS 기준)")

                try:
                    summary_line = _make_intuitive_summary(scores, threshold=1.0)
                    st.info(summary_line)
                except Exception:
                    pass

                # --- Per-meal breakdown ---
                try:
                    if source_mode == "오늘 기록 사용":
                        _df_today = _today_food_log_df()
                        df_meal = _per_meal_breakdown(food_db, _df_today)
                        if not df_meal.empty:
                            st.markdown("#### 🍽️ 식사별 보충 포인트 (오늘)")
                            for label in ["아침","점심","저녁","간식"]:
                                sub = df_meal[df_meal["시간대"] == label]
                                if sub.empty:
                                    continue
                                st.markdown(f"**{label}**")
                                st.dataframe(
                                    sub[["시각","입력항목","매칭식품","채워진태그","직관설명"]]
                                      .rename(columns={
                                          "시각":"시간", "입력항목":"먹은 것",
                                          "매칭식품":"매칭", "채워진태그":"태그", "직관설명":"한줄설명"
                                      }),
                                    use_container_width=True, height=min(300, 60+28*len(sub))
                                )
                                uniq_benefits = []
                                for s in sub["직관설명"].tolist():
                                    for part in [x.strip() for x in s.split("·")]:
                                        if part and part not in uniq_benefits:
                                            uniq_benefits.append(part)
                                if uniq_benefits:
                                    st.caption("보충된 포인트: " + " · ".join(uniq_benefits[:6]))
                except Exception:
                    pass

                # 다음 식사 제안
                st.markdown("#### 🍽️ 다음 식사 제안 (3가지)")
                recent_items = []
                try:
                    if diversity_n > 0:
                        r = _ensure_log()
                        r = r[r["type"] == "food"].copy()
                        if not r.empty:
                            r["date"] = r["date"].astype(str)
                            r["time"] = r["time"].astype(str)
                            recent_df = r.sort_values(["date", "time"]).tail(diversity_n * 5)
                            recent_items = (recent_df["food_norm"].fillna("") + "|" + recent_df["item"].fillna("")).tolist()
                            recent_items = [x.split("|")[0] for x in recent_items if x]
                except Exception:
                    recent_items = []

                seed = hash(("quick-eval", text_in)) % (10**9)
                favor_tags = missing
                cols = st.columns(3)
                for i in range(3):
                    try:
                        try:
                            rng = random.Random(seed + i)
                            title, meal, explain = gen_meal(
                                food_db, include_caution, mode="기본",
                                recent_items=recent_items, favor_tags=favor_tags,
                                rng=rng, user_rules=user_rules_local, allow_rare=False
                            )
                        except Exception:
                            df2 = food_db.copy()
                            if not include_caution:
                                df2 = df2[df2["등급"] != "Caution"]
                            pool = df2["식품"].tolist()
                            rng = random.Random(seed + i)
                            meal = rng.sample(pool, k=min(3, len(pool))) if len(pool) >= 3 else pool
                            title, explain = "다음 식사 제안", ("부족 태그 보완 중심: " + ", ".join(favor_tags)) if favor_tags else ""
                        with cols[i]:
                            st.markdown(f"**{title} #{i+1}**")
                            st.write(" / ".join(meal))
                            if favor_tags:
                                why = [f"· {t}: {NUTRIENT_TIPS.get(t, '')}" for t in favor_tags[:2]]
                                st.caption("보완 포인트:\n" + "\n".join(why))
                            elif explain:
                                st.caption(explain)
                    except Exception as e:
                        st.error(f"제안 생성 실패: {e}")
            except Exception as e:
                st.error(f"분석 실패: {e}")
except Exception:
    pass

# ==== [END ADDON] =============================================================

# ==== [END ADDON] =============================================================

# =============================
# Compatibility Layer (app → v9)
# =============================
# 이 섹션은 기존 app.py에서 사용하던 헬퍼/함수명을 v9 스타일 내부 구현으로 연결합니다.
# v9의 구조/스타일을 유지하면서, app 코드의 호출부를 최대한 그대로 동작하게 만듭니다.

try:
    import pandas as _pd
    import os as _os
    from datetime import datetime as _dt
except Exception:
    pass

def load_nutrient_dict(path: str = "data/nutrient_dict.csv"):
    """
    app.py 호환: 별도의 nutrient_dict.csv 를 로드하려는 시도.
    v9에서는 내장된 NUTRIENT_TIPS / NUTRIENT_SOURCES 를 사용하므로,
    파일이 존재하면 읽어가고, 없으면 v9의 사전으로 graceful fallback 합니다.
    """
    try:
        if _os.path.exists(path):
            df = _pd.read_csv(path)
            # 기대 포맷: key, short_tip, long_tip (있다면)
            tips = {}
            tips_long = {}
            for _, row in df.iterrows():
                key = str(row.get("key") or "").strip()
                if not key:
                    continue
                short_tip = str(row.get("short_tip") or "").strip()
                long_tip = str(row.get("long_tip") or short_tip).strip()
                tips[key] = short_tip or NUTRIENT_TIPS.get(key, "")
                tips_long[key] = long_tip or NUTRIENT_TIPS.get(key, "")
            return tips, tips_long
    except Exception:
        pass
    # fallback
    return dict(NUTRIENT_TIPS), dict(NUTRIENT_TIPS)

def save_log(df):
    """
    app.py 호환: 전체 로그 DataFrame을 저장.
    v9는 add_log_row 등 단위 추가 위주지만, 여기서는 전체 저장도 지원합니다.
    """
    try:
        if isinstance(LOG_PATH, str):
            df.to_csv(LOG_PATH, index=False)
        else:
            # LOG_PATH가 경로 객체인 경우
            _pd.DataFrame(df).to_csv(str(LOG_PATH), index=False)
    except Exception:
        # 실패 시에도 앱이 멈추지 않도록 함
        pass

def today_df():
    """
    app.py 호환: 오늘 날짜의 로그를 DataFrame으로 반환.
    v9 내부 함수 _today_food_log_df() 호출을 우선 시도하고, 없으면 직접 필터링.
    """
    try:
        return _today_food_log_df()
    except Exception:
        try:
            if isinstance(LOG_PATH, str) and _os.path.exists(LOG_PATH):
                df = _pd.read_csv(LOG_PATH)
                today = _dt.now().strftime("%Y-%m-%d")
                if "date" in df.columns:
                    return df[df["date"].astype(str) == today].copy()
            return _pd.DataFrame()
        except Exception:
            return _pd.DataFrame()


def _parse_tags_flexible(v):
    """
    태그(영양) 컬럼이 list 또는 문자열("단백질, 식이섬유") 등 다양한 형태로 올 수 있어
    안전하게 list[str]로 변환합니다.
    """
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v)
    # 콤마/슬래시/공백 구분자를 모두 허용
    parts = re.split(r"[,\u3001/;|]+|\s{2,}", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 괄호나 해시 제거
        p = re.sub(r"[#\[\]\(\)]+", "", p).strip()
        if p:
            out.append(p)
    return out


# === User-friendly nutrient mapping & synonyms ===
NUTRIENT_SYNONYMS = dict(globals().get("NUTRIENT_SYNONYMS", {}))
NUTRIENT_FRIENDLY = dict(globals().get("NUTRIENT_FRIENDLY", {}))
NUTRIENT_DEFAULT_EXAMPLES = dict(globals().get("NUTRIENT_DEFAULT_EXAMPLES", {}))

# Canonical → friendly label (emoji + Korean)
NUTRIENT_FRIENDLY.update({
    "Protein": "단백질 🍗",
    "Fiber": "식이섬유 🥦",
    "ComplexCarb": "복합탄수화물 🍚",
    "HealthyFat": "건강한 지방 🥑",
    "Omega3": "오메가-3 🐟",
    "A": "비타민 A 🥕",
    "B": "비타민 B군 🍞",
    "C": "비타민 C 🍊",
    "D": "비타민 D ☀️",
    "E": "비타민 E 🥜",
    "K": "비타민 K 🥬",
    "Ca": "칼슘 🦴",
    "Mg": "마그네슘 😌",
    "Fe": "철분 💪",
    "K_potassium": "칼륨(부종/혈압) 🧂↘️",
})

# Abbreviation & alias → canonical key
NUTRIENT_SYNONYMS.update({
    "Protein":"Protein", "단백질":"Protein",
    "Fiber":"Fiber", "식이섬유":"Fiber",
    "ComplexCarb":"ComplexCarb","복합탄수화물":"ComplexCarb","slowcarb":"ComplexCarb","slow_carb":"ComplexCarb",
    "HealthyFat":"HealthyFat","건강한지방":"HealthyFat","goodfat":"HealthyFat",
    "Omega3":"Omega3","오메가3":"Omega3","오메가-3":"Omega3","EPA/DHA":"Omega3",
    "A":"A","비타민A":"A",
    "B":"B","비타민B":"B","비타민B군":"B",
    "C":"C","비타민C":"C",
    "D":"D","비타민D":"D",
    "E":"E","비타민E":"E",
    "K":"K","비타민K":"K",
    "Ca":"Ca","칼슘":"Ca",
    "Mg":"Mg","마그네슘":"Mg",
    "Fe":"Fe","철":"Fe","철분":"Fe",
    "K_potassium":"K_potassium","칼륨":"K_potassium","Potassium":"K_potassium",
})

# Fallback examples when none in CSV / food_db
NUTRIENT_DEFAULT_EXAMPLES.update({
    "Protein": ["닭가슴살","두부","연어","계란","그릭요거트"],
    "Fiber": ["현미밥","귀리","사과","브로콜리","렌틸콩"],
    "ComplexCarb": ["현미밥","귀리","통밀빵","고구마","퀴노아"],
    "HealthyFat": ["아보카도","올리브유","아몬드","호두","참치"],
    "Omega3": ["연어","고등어","정어리","호두","치아시드"],
    "A": ["당근","호박","시금치","케일","간"],
    "B": ["현미","귀리","달걀","버섯","돼지고기"],
    "C": ["키위","파프리카","브로콜리","귤","딸기"],
    "D": ["연어","계란","버섯(일광건조)","강화우유"],
    "E": ["아몬드","해바라기씨","올리브유","아보카도"],
    "K": ["케일","시금치","브로콜리","상추"],
    "Ca": ["두부","요거트","멸치","브로콜리","우유"],
    "Mg": ["시금치","현미","아몬드","호두","다크초콜릿"],
    "Fe": ["소간","시금치","홍합","렌틸콩","강화시리얼"],
    "K_potassium": ["바나나","아보카도","감자","고구마","시금치"],
})

def _canon_key(k: str):
    k = str(k or "").strip()
    return NUTRIENT_SYNONYMS.get(k, k)

def _friendly_label(k: str):
    key = _canon_key(k)
    return NUTRIENT_FRIENDLY.get(key, key)


def _lookup_tip(key: str):
    """BENEFIT_MAP 우선, 없으면 NUTRIENT_TIPS. 한/영 양쪽 키 모두 시도."""
    k = _canon_key(key)
    # canonical 먼저
    v = (BENEFIT_MAP.get(k) or NUTRIENT_TIPS.get(k) or NUTRIENT_TIPS_LONG.get(k) if 'NUTRIENT_TIPS_LONG' in globals() else None)
    if v: return v
    # 원 키(한글일 수 있음)도 시도
    return (BENEFIT_MAP.get(key) or NUTRIENT_TIPS.get(key) or (NUTRIENT_TIPS_LONG.get(key) if 'NUTRIENT_TIPS_LONG' in globals() else "")) or ""

def _harmonize_mappings():
    """현재 로딩된 사전의 키들을 한/영 모두로 복제하여 조회 실패를 방지."""
    try:
        # 복사본에서 순회
        for dname in ["NUTRIENT_TIPS", "NUTRIENT_TIPS_LONG", "BENEFIT_MAP", "NUTRIENT_EXAMPLES"]:
            if dname not in globals():
                continue
            d = globals()[dname]
            if not isinstance(d, dict):
                continue
            add_items = {}
            for k, v in list(d.items()):
                ck = _canon_key(k)
                if ck and ck not in d and v not in (None, ""):
                    add_items[ck] = v
                # 반대로 한글 키도 확보 (친절 라벨에서 한국어 조회 시)
                # 간단 매핑: canonical → friendly 라벨에서 한국어 추출
                try:
                    # _friendly_label(ck) 반환값이 "비타민 C 🍊" 같은 형태일 수 있으므로 한글만 쓰지 않고 ck 자체 사용
                    pass
                except Exception:
                    pass
            d.update(add_items)
    except Exception:
        pass

# 실행 시점에 한 번 정합화
_harmonize_mappings()
def _example_foods_for(k: str, limit=6):
    key = _canon_key(k)
    ex = []
    try:
        ex = (NUTRIENT_EXAMPLES.get(key) if 'NUTRIENT_EXAMPLES' in globals() else None) or []
    except Exception:
        ex = []
    if not ex:
        ex = NUTRIENT_DEFAULT_EXAMPLES.get(key, [])
    return list(dict.fromkeys(ex))[:limit]
def _benefit_from_tag(tag):
    """
    단일 태그에서 베네핏 한줄을 생성.
    1) BENEFIT_MAP → 2) NUTRIENT_TIPS → 3) 동의어 매핑(_nut_ko/_nut_en) → 4) 기본 문구
    """
    if 'BENEFIT_MAP' in globals() and tag in BENEFIT_MAP and BENEFIT_MAP.get(tag):
        return BENEFIT_MAP.get(tag)
    if 'NUTRIENT_TIPS' in globals() and tag in NUTRIENT_TIPS and NUTRIENT_TIPS.get(tag):
        return NUTRIENT_TIPS.get(tag)
    # 동의어 시도
    try:
        for alt in {tag, _nut_ko(tag), _nut_en(tag)}:
            if alt and 'BENEFIT_MAP' in globals() and BENEFIT_MAP.get(alt):
                return BENEFIT_MAP[alt]
            if alt and 'NUTRIENT_TIPS' in globals() and NUTRIENT_TIPS.get(alt):
                return NUTRIENT_TIPS[alt]
    except Exception:
        pass
    return ""
def _to_tags(text):
    """
    app.py 호환: 자유 텍스트에서 간단한 태그 추출.
    v9의 토큰/스코어링 로직이 더 풍부하므로, 선행 사용 후 보조적으로 키워드 매핑을 적용.
    """
    try:
        toks = split_free_text(text)
    except Exception:
        toks = []
    tags = set()
    for t in toks:
        base = t.strip().lower()
        if not base:
            continue
        # 간단 매핑: 커피/차/과일 등
        for k, v in KEYWORD_MAP.items():
            if k.lower() in base:
                tags.add(v)
    return sorted(tags)
