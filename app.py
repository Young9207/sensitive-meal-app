# -*- coding: utf-8 -*-
"""
Nutrition Coach — Operational (fixed CSV paths, no uploader)
Run:
    streamlit run app_full_operational.py
Data folder:
    ./data/nutrient_dict.csv  (영양사전)
    ./data/food_db.csv        (식품 DB)
    ./data/food_log.csv       (식사 기록)
"""
import re, ast, random, os
from typing import List, Tuple, Dict, Any
from datetime import datetime, date, time

import pandas as pd
import streamlit as st
from difflib import get_close_matches
from pathlib import Path

# ---- Fixed data paths ----
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
NUTRI_CSV = DATA_DIR / "nutrient_dict.csv"
FOOD_CSV  = DATA_DIR / "food_db.csv"
LOG_CSV   = DATA_DIR / "food_log.csv"

# -----------------------------
# Defaults (used if CSV missing columns/keys)
# -----------------------------
CORE_NUTRIENTS = [
    "단백질", "식이섬유", "철", "칼슘", "마그네슘", "칼륨",
    "오메가3", "비타민A", "비타민B", "비타민C", "비타민D", "비타민E",
    "저당", "저염", "건강한지방"
]
ESSENTIALS = ["단백질", "식이섬유", "비타민C", "칼슘"]

DEFAULT_TIPS = {
    "단백질": "근육·포만감", "식이섬유": "장건강·포만감·혈당완화",
    "철": "피로·집중", "칼슘": "뼈·치아", "마그네슘": "긴장완화·수면",
    "칼륨": "붓기·혈압", "오메가3": "심혈관·염증완화",
    "비타민A": "눈·피부", "비타민B": "에너지대사", "비타민C": "면역·철흡수",
    "비타민D": "뼈·면역", "비타민E": "항산화·피부", "저당": "혈당완화",
    "저염": "붓기·혈압", "건강한지방": "포만감·흡수", "저지방": "가벼움·단백질확보"
}
DEFAULT_LONG = {
    "단백질": "근육 유지·회복, 포만감 유지에 도움.",
    "식이섬유": "배변 규칙성, 포만감, 혈당 급상승 완화.",
    "철": "산소 운반으로 피로/어지럼 완화, 비타민 C와 함께 흡수↑",
    "칼슘": "뼈·치아 건강, 근육/신경 기능.",
    "마그네슘": "근육 이완, 수면·긴장 완화, 에너지 대사.",
    "칼륨": "나트륨 배출로 붓기·혈압 조절.",
    "오메가3": "심혈관·뇌 건강, 염증 균형.",
    "비타민A": "야간 시력·피부·점막 보호.",
    "비타민B": "에너지 생성·피로 완화(복합군).",
    "비타민C": "면역, 철 흡수, 항산화.",
    "비타민D": "칼슘 흡수·뼈 건강, 면역 조절.",
    "비타민E": "항산화(세포 보호), 피부 컨디션.",
    "저당": "식후 혈당 변동 완화.", "저염": "붓기 완화·혈압 관리.",
    "건강한지방": "포만감·지용성 비타민 흡수."
}
DEFAULT_SOURCES = {
    "단백질": ["닭가슴살","두부","연어","계란"],
    "식이섬유": ["현미밥","귀리","브로콜리","양배추"],
    "칼슘": ["두부","요거트","브로콜리","아몬드"],
    "비타민D": ["계란","연어","버섯(일광)"],
    "비타민C": ["브로콜리","양배추","키위","파프리카"],
    "오메가3": ["연어","고등어","호두"],
    "칼륨": ["아보카도","바나나","감자","시금치"],
    "마그네슘": ["현미밥","시금치","견과류"],
    "비타민E": ["올리브유","아몬드","아보카도"],
    "비타민A": ["당근","시금치","호박"],
    "비타민B": ["버섯","통곡물","달걀"],
    "건강한지방": ["올리브유","아보카도","견과류"]
}

# Will be filled from CSV
NUTRIENT_TIPS = dict(DEFAULT_TIPS)        # short line
NUTRIENT_TIPS_LONG = dict(DEFAULT_LONG)   # long
BENEFIT_MAP = dict(DEFAULT_TIPS)          # alias: label for one-liner
NUTRIENT_SOURCES = dict(DEFAULT_SOURCES)

# -----------------------------
# IO & helpers
# -----------------------------
def _to_tags(val):
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    parts = [x.strip() for x in re.split(r'[\/,\|\;]+', s) if x.strip()]
    out = []
    for t in parts:
        t2 = t.strip().strip('"').strip("'")
        if t2 == "식이 섬유": t2 = "식이섬유"
        out.append(t2)
    return out

def load_food_db() -> pd.DataFrame:
    if FOOD_CSV.exists():
        df = pd.read_csv(FOOD_CSV)
        if "태그(영양)" in df.columns:
            df["태그(영양)"] = df["태그(영양)"].apply(_to_tags)
        return df
    return pd.DataFrame(columns=["식품","등급","태그(영양)"])

def load_nutrient_dict():
    global NUTRIENT_TIPS, NUTRIENT_TIPS_LONG, BENEFIT_MAP, NUTRIENT_SOURCES
    if not NUTRI_CSV.exists():
        return
    try:
        df = pd.read_csv(NUTRI_CSV)
    except Exception:
        return
    # expected columns
    for _, r in df.iterrows():
        k = str(r.get("영양소") or "").strip()
        if not k: continue
        s = str(r.get("한줄설명") or "").strip()
        l = str(r.get("자세한설명") or "").strip()
        b = str(r.get("혜택라벨(요약)") or "").strip()
        src = str(r.get("대표식품(쉼표로구분)") or "").strip()
        if s: NUTRIENT_TIPS[k] = s
        if l: NUTRIENT_TIPS_LONG[k] = l
        if b: BENEFIT_MAP[k] = b
        if src:
            NUTRIENT_SOURCES[k] = [x.strip() for x in src.split(",") if x.strip()]

def ensure_log() -> pd.DataFrame:
    if LOG_CSV.exists():
        try:
            df = pd.read_csv(LOG_CSV)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["date","time","item","food_norm","qty"])

def save_log(df: pd.DataFrame):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_CSV, index=False)

def match_food(name: str, df_food: pd.DataFrame) -> Tuple[str, bool]:
    names = df_food["식품"].tolist() if "식품" in df_food.columns else []
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

def split_free_text(text: str) -> List[str]:
    if not text: return []
    return [p.strip() for p in re.split(r"[,|\n]+", text) if p.strip()]

def parse_qty(token: str) -> Tuple[str, float]:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*$", token)
    if m:
        qty = float(m.group(1))
        name = token[:m.start()].strip()
        return name, qty
    return token.strip(), 1.0

def score_tokens(free_text: str, df_food: pd.DataFrame):
    tokens = split_free_text(free_text)
    score = {k: 0.0 for k in CORE_NUTRIENTS}
    rows = []
    for tok in tokens:
        name_raw, qty = parse_qty(tok)
        mapped, matched = match_food(name_raw, df_food)
        tags = []
        if matched and "태그(영양)" in df_food.columns:
            try:
                rec = df_food[df_food["식품"] == mapped].iloc[0]
                tags = _to_tags(rec.get("태그(영양)", []))
            except Exception:
                tags = []
        for t in tags:
            if t in score:
                score[t] += float(qty or 1.0)
        rows.append({"식품": name_raw, "매칭": mapped if matched else name_raw, "수량": qty, "태그": ", ".join(tags)})
    return score, pd.DataFrame(rows)

def make_intuitive_summary(scores: Dict[str, float], thr: float=1.0) -> str:
    filled, low = [], []
    ordered = list(ESSENTIALS) + [k for k in BENEFIT_MAP.keys() if k not in ESSENTIALS]
    for k in ordered:
        v = float(scores.get(k, 0) or 0)
        b = BENEFIT_MAP.get(k)
        if not b: continue
        if v >= thr:
            if b not in filled: filled.append(b)
        else:
            if b not in low: low.append(b)
    L = " · ".join(filled[:3])
    R = " · ".join(low[:3])
    if L and R: return f"오늘 한 줄 요약: {L}는 꽤 채워졌고, {R}는 보충이 필요해요."
    if L: return f"오늘 한 줄 요약: {L}는 잘 챙겨졌어요."
    if R: return f"오늘 한 줄 요약: {R} 보충이 필요해요."
    return "오늘 한 줄 요약: 분석할 항목이 없어요."

def today_df(df_log: pd.DataFrame) -> pd.DataFrame:
    if df_log.empty: return df_log.copy()
    df = df_log.copy()
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    except Exception:
        return df.iloc[0:0]
    today = date.today()
    df = df[df["date"] == today]
    # time-of-day
    def _label(tstr):
        try:
            h = pd.to_datetime(str(tstr)).hour
        except Exception:
            return "간식"
        if 5 <= h < 11: return "아침"
        if 11 <= h < 16: return "점심"
        if 16 <= h < 21: return "저녁"
        return "간식"
    df["시간대"] = df["time"].apply(_label)
    return df

def per_meal_breakdown(df_food: pd.DataFrame, df_today: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_today.iterrows():
        raw = str(r.get("item") or "").strip()
        if not raw:
            continue
        mapped, matched = match_food(raw, df_food)
        tags = []
        if matched and "태그(영양)" in df_food.columns:
            try:
                rec = df_food[df_food["식품"] == mapped].iloc[0]
                tags = _to_tags(rec.get("태그(영양)", []))
            except Exception:
                tags = []
        benefits = []
        for t in tags:
            b = BENEFIT_MAP.get(t) or NUTRIENT_TIPS.get(t) or ""
            if b and b not in benefits:
                benefits.append(b)
        rows.append({
            "시간대": r.get("시간대",""),
            "시간": r.get("time",""),
            "먹은 것": raw,
            "매칭": mapped if matched else raw,
            "태그": ", ".join(tags[:5]),
            "한줄설명": " · ".join(benefits[:3])
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["시간대","시간"])
    return out

def gen_meal(df_food: pd.DataFrame, missing_tags: List[str], k: int=3) -> List[List[str]]:
    df = df_food.copy()
    picks_all = []
    import random as _r
    for i in range(3):
        pool = []
        for _, r in df.iterrows():
            tags = _to_tags(r.get("태그(영양)", []))
            overlap = len(set(tags) & set(missing_tags))
            pool.append((overlap, r["식품"]))
        pool.sort(key=lambda x: (-x[0], x[1]))
        cand = [n for ov, n in pool if ov > 0] or df["식품"].tolist()
        picks = _r.sample(cand, k=min(k, len(cand))) if len(cand) >= k else cand
        picks_all.append(picks)
    return picks_all

# -----------------------------
# Boot: load CSVs
# -----------------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
load_nutrient_dict()
food_db = load_food_db()
log_df = ensure_log()

# -----------------------------
# UI — Tabs
# -----------------------------
st.set_page_config(page_title="Nutrition Coach (Ops)", layout="wide")
st.title("🥗 Nutrition Coach — 운영용")

tabs = st.tabs(["📝 기록", "📊 요약/제안", "⚡ 즉석 평가", "📤 내보내기", "🛠 관리"])

# ---- 기록 ----
with tabs[0]:
    st.subheader("식사 기록 추가")
    col1, col2, col3 = st.columns([1.2,1,1])
    with col1:
        item = st.text_input("먹은 것 (예: 대구구이, 브로콜리 등)")
    with col2:
        d = st.date_input("날짜", value=date.today(), format="YYYY-MM-DD")
    with col3:
        t = st.time_input("시간", value=pd.Timestamp.now().time())
    qty = st.number_input("수량(선택)", min_value=0.0, value=1.0, step=0.5)
    if st.button("기록 추가", type="primary", use_container_width=True):
        log_df = ensure_log()
        new = pd.DataFrame([{"date": d, "time": t, "item": item, "food_norm": "", "qty": qty}])
        log_df = pd.concat([log_df, new], ignore_index=True)
        save_log(log_df)
        st.success("추가되었습니다. (data/food_log.csv)")
    st.divider()
    st.subheader("최근 기록")
    st.dataframe(ensure_log().tail(50), use_container_width=True, height=300)

# ---- 요약/제안 ----
with tabs[1]:
    st.subheader("오늘 요약 (당일 기준)")
    df_today = today_df(ensure_log())
    if df_today.empty:
        st.info("오늘 기록이 없습니다.")
    else:
        # Build text for scoring
        text_in = ", ".join(df_today["item"].dropna().astype(str).tolist())
        scores, parsed = score_tokens(text_in, food_db)
        st.markdown("**파싱 결과**")
        st.dataframe(parsed, use_container_width=True, height=240)

        st.markdown("**태그 점수 + 한줄설명**")
        score_df = (
            pd.DataFrame([scores]).T.reset_index().rename(columns={"index":"영양소", 0:"점수"})
            .sort_values("점수", ascending=False)
        )
        score_df["한줄설명"] = score_df["영양소"].map(lambda x: NUTRIENT_TIPS.get(x, ""))
        st.dataframe(score_df, use_container_width=True, height=320)

        # Missing essentials
        missing = [n for n in ESSENTIALS if scores.get(n, 0) < 1]
        if missing:
            st.warning("부족 태그: " + ", ".join(missing))
        st.info(make_intuitive_summary(scores, thr=1.0))

        # Per meal breakdown
        st.markdown("**🍽️ 식사별 보충 포인트 (오늘)**")
        meal_df = per_meal_breakdown(food_db, df_today)
        if meal_df.empty:
            st.info("표시할 식사 기록이 없습니다.")
        else:
            for label in ["아침","점심","저녁","간식"]:
                sub = meal_df[meal_df["시간대"] == label]
                if sub.empty: continue
                st.markdown(f"**{label}**")
                st.dataframe(sub, use_container_width=True, height=min(300, 60+28*len(sub)))
                # badges
                uniq = []
                for s in sub["한줄설명"].tolist():
                    for part in [x.strip() for x in s.split("·")]:
                        if part and part not in uniq:
                            uniq.append(part)
                if uniq:
                    st.caption("보충된 포인트: " + " · ".join(uniq[:6]))

        # Suggestions
        st.markdown("**🍽️ 다음 식사 제안**")
        cols = st.columns(3)
        picks3 = gen_meal(food_db, missing_tags=missing, k=3)
        for i, picks in enumerate(picks3):
            with cols[i]:
                st.write(f"제안 #{i+1}: " + " / ".join(picks))

        # Glossary
        with st.expander("📘 영양소 한눈 요약 (CSV 기반)", expanded=False):
            df_gloss = pd.DataFrame([
                {"영양소": k,
                 "무엇에 좋은가(쉽게)": NUTRIENT_TIPS_LONG.get(k, NUTRIENT_TIPS.get(k, "")),
                 "대표 식품": ", ".join(NUTRIENT_SOURCES.get(k, [])[:4])}
                for k in CORE_NUTRIENTS if (k in NUTRIENT_TIPS or k in NUTRIENT_TIPS_LONG)
            ])
            st.dataframe(df_gloss, use_container_width=True, height=360)

# ---- 즉석 평가 ----
with tabs[2]:
    st.subheader("붙여넣기 분석 (저장 없이)")
    sample = "쌀밥1, 대구구이1, 브로콜리1, 올리브유0.5"
    text_in = st.text_area("식단 텍스트", height=120, placeholder=sample)
    if st.button("분석하기", type="primary"):
        scores, parsed = score_tokens(text_in, food_db)
        st.markdown("**파싱 결과**")
        st.dataframe(parsed, use_container_width=True, height=240)
        st.markdown("**태그 점수 + 한줄설명**")
        score_df = (
            pd.DataFrame([scores]).T.reset_index().rename(columns={"index":"영양소", 0:"점수"})
            .sort_values("점수", ascending=False)
        )
        score_df["한줄설명"] = score_df["영양소"].map(lambda x: NUTRIENT_TIPS.get(x, ""))
        st.dataframe(score_df, use_container_width=True, height=320)
        st.info(make_intuitive_summary(scores, thr=1.0))

# ---- 내보내기 ----
with tabs[3]):
    st.subheader("CSV 내보내기")
    df_all = ensure_log()
    csv_all = df_all.to_csv(index=False).encode("utf-8")
    st.download_button("식사 기록 CSV 다운로드", data=csv_all, file_name="food_log.csv", mime="text/csv")
    # 일일 요약(오늘만)
    df_today = today_df(df_all)
    text_in = ", ".join(df_today["item"].dropna().astype(str).tolist())
    scores, _ = score_tokens(text_in, food_db) if len(text_in) else ({}, pd.DataFrame())
    daily = pd.DataFrame([{"date": date.today().isoformat(), **scores}])
    st.download_button("오늘 요약 CSV 다운로드", data=daily.to_csv(index=False).encode("utf-8"),
                       file_name="daily_summary.csv", mime="text/csv")

# ---- 관리 ----
with tabs[4]:
    st.subheader("데이터 파일 위치")
    st.code(f"영양사전: {NUTRI_CSV}\n식품DB:   {FOOD_CSV}\n기록:     {LOG_CSV}")
    if st.button("오늘 기록 삭제"):
        df = ensure_log()
        df = df[df["date"] != date.today().isoformat()]
        save_log(df)
        st.success("오늘 날짜 기록을 삭제했습니다.")
    if st.button("전체 기록 삭제", type="secondary"):
        save_log(pd.DataFrame(columns=["date","time","item","food_norm","qty"]))
        st.warning("전체 기록을 초기화했습니다.")
