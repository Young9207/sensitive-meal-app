# -*- coding: utf-8 -*-
"""
app_v5.py — Instant Diet Evaluator (fixed)
-----------------------------------------
- Robust tag parsing (_to_tags)
- Today-only analysis option (logs)
- One-line intuitive daily summary
- Per-meal breakdown: what boosted which nutrients (with simple benefits)
- Nutrient tips + glossary (plain-language)

Run:
    streamlit run app_v5.py
"""
import re
import ast
import random
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st
from difflib import get_close_matches

# -----------------------------
# Minimal in-app "DB" (demo)
# -----------------------------
CORE_NUTRIENTS = [
    "단백질", "식이섬유", "철", "칼슘", "마그네슘", "칼륨",
    "오메가3", "비타민A", "비타민B", "비타민C", "비타민D", "비타민E",
    "저당", "저염", "건강한지방"
]
ESSENTIALS = ["단백질", "식이섬유", "비타민C", "칼슘"]

NUTRIENT_TIPS: Dict[str, str] = {
    "단백질": "근육 유지와 포만감에 좋아요—식사 후 허기를 줄여줘요.",
    "식이섬유": "배변 리듬과 포만감, 혈당 급상승을 완화해요.",
    "철": "피로감 줄이고 집중에 도움—혈액 산소 운반에 필수예요.",
    "칼슘": "뼈·치아 건강의 기본—근육 수축에도 필요해요.",
    "마그네슘": "긴장 완화와 수면·근육 기능에 도움을 줘요.",
    "칼륨": "나트륨 배출을 도와 붓기·혈압 관리에 유리해요.",
    "오메가3": "심혈관 건강과 염증 균형에 도움—등푸른 생선에 많아요.",
    "비타민A": "눈·피부 점막 보호—색 진한 채소에 풍부해요.",
    "비타민B": "에너지 대사 서포트—피로감 완화에 도움.",
    "비타민C": "면역과 철 흡수에 도움—가열 덜 한 채소·과일로.",
    "비타민D": "뼈 건강과 면역에 도움—햇빛·계란·생선에도 있어요.",
    "비타민E": "세포 보호(항산화)와 피부 컨디션에 도움.",
    "저당": "식후 혈당 출렁임을 줄이는 선택이에요.",
    "저염": "붓기·혈압 관리에 유리—가공식품 소금 체크!",
    "건강한지방": "포만감·지용성 비타민 흡수에 도움(아보카도·견과).",
    "저지방": "열량 대비 단백질을 채우기 좋고 부담이 덜해요."
}
NUTRIENT_TIPS_LONG: Dict[str, str] = {
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
BENEFIT_MAP: Dict[str, str] = {
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
NUTRIENT_SOURCES: Dict[str, List[str]] = {
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

# -----------------------------
# Helpers
# -----------------------------
def _to_tags(val):
    """Normalize a '태그(영양)' cell into a list of clean tag strings."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    # Try JSON/python-like list: "['단백질','칼슘']" or '["단백질", "칼슘"]'
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    # Fallback: split by common separators
    parts = [x.strip() for x in re.split(r'[\/,\|\;]+', s) if x.strip()]
    out = []
    for t in parts:
        t2 = t.strip().strip('"').strip("'")
        if t2 == "식이 섬유":
            t2 = "식이섬유"
        out.append(t2)
    return out

def split_free_text(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[,|\n]+", text)
    return [p.strip() for p in parts if p.strip()]

def parse_qty(token: str) -> Tuple[str, float]:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*$", token)
    if m:
        qty = float(m.group(1))
        name = token[: m.start()].strip()
        return name, qty
    return token.strip(), 1.0

def contains_any(text: str, keywords: List[str]) -> bool:
    text = (text or "").lower()
    for k in keywords or []:
        if k.lower() in text:
            return True
    return False

def match_food(name: str, df_food: pd.DataFrame) -> Tuple[str, bool]:
    names = df_food["식품"].tolist()
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

def ensure_log() -> pd.DataFrame:
    """Stub for demo. Replace to connect with your real diary/log storage."""
    return pd.DataFrame(columns=["type", "date", "time", "food_norm", "item"])

def tokens_from_today_log() -> List[str]:
    import datetime as _dt
    df = ensure_log()
    if df is None or df.empty:
        return []
    today = _dt.datetime.now().date()
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    except Exception:
        return []
    df_today = df[(df["type"] == "food") & (df["date"] == today)].copy()
    now_time = _dt.datetime.now().time()
    try:
        df_today["time"] = pd.to_datetime(df_today["time"].astype(str), errors="coerce").dt.time
        df_today = df_today[df_today["time"].isna() | (df_today["time"] <= now_time)]
    except Exception:
        pass
    toks = []
    for _, r in df_today.iterrows():
        token = (str(r.get("item") or "")).strip() or (str(r.get("food_norm") or "")).strip()
        if token:
            toks.append(token)
    return toks

def today_food_log_df() -> pd.DataFrame:
    """Return today's log with time-of-day labels for per-meal breakdown."""
    import datetime as _dt
    df = ensure_log()
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

def per_meal_breakdown(df_food: pd.DataFrame, df_today: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_today.iterrows():
        raw = str(r.get("item") or "").strip() or str(r.get("food_norm") or "").strip()
        if not raw:
            continue
        mapped, matched = match_food(raw, df_food)
        tags = []
        if matched:
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
            "시간대": r.get("시간대", ""),
            "시각": r.get("_dt"),
            "먹은 것": raw,
            "매칭": mapped if matched else raw,
            "태그": ", ".join(tags[:5]),
            "한줄설명": " · ".join(benefits[:3])
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["시간대", "시각"])
    return out

def score_tokens(free_text: str, df_food: pd.DataFrame, user_rules: Dict[str, Any]):
    tokens = split_free_text(free_text)
    rows = []
    score = {k: 0.0 for k in CORE_NUTRIENTS}
    for tok in tokens:
        name_raw, qty = parse_qty(tok)
        name_norm = (name_raw or "").strip()
        if not name_norm:
            continue
        if contains_any(name_norm, user_rules.get("avoid_keywords", [])):
            rows.append({
                "식품": name_raw, "정규화": name_norm, "수량": qty,
                "등급": "Avoid", "사유": "개인 회피리스트", "태그(영양)": []
            })
            continue
        mapped, matched = match_food(name_norm, df_food)
        tags, grade, flags = [], "Safe", ""
        if matched:
            rec = df_food[df_food["식품"] == mapped].iloc[0]
            grade = rec.get("등급", "Safe")
            tags = _to_tags(rec.get("태그(영양)", []))
            if contains_any(name_norm, user_rules.get("allow_keywords", [])) and grade != "Avoid":
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
    items_df = pd.DataFrame(rows)
    return score, items_df

def gen_meal(df_food: pd.DataFrame, include_caution: bool,
             recent_items: List[str], favor_tags: List[str], rng: random.Random) -> Tuple[str, List[str], str]:
    df = df_food.copy()
    if not include_caution:
        df = df[df["등급"] != "Caution"]
    # prioritize by favor_tags
    pool = []
    for _, r in df.iterrows():
        tags = _to_tags(r.get("태그(영양)", []))
        overlap = len(set(tags) & set(favor_tags or []))
        pool.append((overlap, r["식품"]))
    pool.sort(key=lambda x: (-x[0], x[1]))
    cand = [name for ov, name in pool if ov > 0] or df["식품"].tolist()
    # avoid recent repeats
    recent_items = set(recent_items or [])
    cand = [x for x in cand if x not in recent_items] or cand
    picks = rng.sample(cand, k=min(3, len(cand))) if len(cand) >= 3 else cand
    explain = ""
    if favor_tags:
        explain = f"부족 태그 보완 중심: {', '.join(favor_tags)}"
    return "다음 식사 제안", picks, explain

def make_intuitive_summary(scores: Dict[str, float], threshold: float = 1.0) -> str:
    filled_benefits, low_benefits = [], []
    ordered = list(ESSENTIALS) + [k for k in BENEFIT_MAP.keys() if k not in ESSENTIALS]
    for k in ordered:
        v = float(scores.get(k, 0) or 0)
        b = BENEFIT_MAP.get(k)
        if not b:
            continue
        if v >= threshold:
            if b not in filled_benefits:
                filled_benefits.append(b)
        else:
            if b not in low_benefits:
                low_benefits.append(b)
    left = " · ".join(filled_benefits[:3]) if filled_benefits else ""
    right = " · ".join(low_benefits[:3]) if low_benefits else ""
    if left and right:
        return f"오늘 한 줄 요약: {left}는 꽤 채워졌고, {right}는 보충이 필요해요."
    if left:
        return f"오늘 한 줄 요약: {left}는 잘 챙겨졌어요."
    if right:
        return f"오늘 한 줄 요약: {right} 보충이 필요해요."
    return "오늘 한 줄 요약: 분석할 항목이 없어요."

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="즉석 식단 평가", layout="wide")
st.title("⚡ 즉석 식단 평가 (Instant Diet Evaluator)")

with st.expander("데모 푸드 DB 보기 / CSV 교체 안내", expanded=False):
    st.write("현재는 데모용 DB를 사용합니다. 원하는 CSV를 업로드하여 교체할 수 있어요.")
    st.dataframe(food_db, use_container_width=True, height=220)
    up = st.file_uploader("CSV로 DB 교체 (식품, 등급, 태그(영양) 컬럼 필요)", type=["csv"])
    if up is not None:
        try:
            df_new = pd.read_csv(up)
            df_new["태그(영양)"] = df_new["태그(영양)"].apply(_to_tags)
            food_db = df_new
            st.success("DB 교체 완료! 아래 분석에 반영됩니다.")
        except Exception as e:
            st.error(f"CSV 파싱 실패: {e}")

with st.sidebar:
    st.header("개인 규칙")
    avoid = st.text_input("회피 키워드(쉼표로 구분)", value="")
    allow = st.text_input("허용 키워드(쉼표로 구분)", value="")
    include_caution = st.checkbox("경계(Caution) 포함해서 제안", value=False)
    diversity_n = st.slider("다양화(최근 N회 중복 회피)", min_value=0, max_value=10, value=5, step=1)

user_rules = {
    "avoid_keywords": [x.strip() for x in avoid.split(",") if x.strip()],
    "allow_keywords": [x.strip() for x in allow.split(",") if x.strip()],
}

st.subheader("입력한 식단을 즉석 분석")
source_mode = st.radio("분석 소스", ["오늘 기록 사용", "직접 입력"], horizontal=True, index=0)
sample = "쌀밥1, 대구구이1, 브로콜리1, 올리브유0.5"
text_in = st.text_area("식단 텍스트 (쉼표 또는 줄바꿈으로 구분, 예: "+sample+")",
                       height=120, placeholder=sample,
                       disabled=(source_mode == "오늘 기록 사용"))

if source_mode == "오늘 기록 사용":
    _toks = tokens_from_today_log()
    if _toks:
        st.caption("오늘 기록에서 불러온 항목: " + ", ".join(_toks))
        text_in = ", ".join(_toks)
    else:
        st.info("오늘 날짜의 음식 기록이 없어요. 직접 입력으로 전환해 주세요.")

col_btn1, col_btn2 = st.columns([1,1])
with col_btn1:
    analyze = st.button("분석하기", type="primary")
with col_btn2:
    clear = st.button("지우기")
if clear:
    text_in = ""
    st.experimental_rerun()

with st.expander("📘 영양소 한눈 요약 (무엇에 좋은가 + 대표 식품)", expanded=False):
    df_gloss = pd.DataFrame([
        {"영양소": k,
         "무엇에 좋은가(쉽게)": NUTRIENT_TIPS_LONG.get(k, NUTRIENT_TIPS.get(k, "")),
         "대표 식품": ", ".join(NUTRIENT_SOURCES.get(k, [])[:4])}
        for k in CORE_NUTRIENTS if k in NUTRIENT_TIPS or k in NUTRIENT_TIPS_LONG
    ])
    st.dataframe(df_gloss, use_container_width=True, height=360)

if analyze:
    try:
        scores, items_df = score_tokens(text_in, food_db, user_rules)

        st.markdown("### 🍱 파싱 결과")
        if items_df.empty:
            st.info("항목이 없습니다. 식단을 입력해 주세요.")
        else:
            st.dataframe(items_df, use_container_width=True, height=280)

        st.markdown("### 🧭 태그 점수 + 한줄 설명")
        score_df = (
            pd.DataFrame([scores])
            .T.reset_index().rename(columns={"index":"영양소", 0:"점수"})
            .sort_values("점수", ascending=False)
        )
        score_df["한줄설명"] = score_df["영양소"].map(lambda x: NUTRIENT_TIPS.get(x, ""))
        st.dataframe(score_df, use_container_width=True, height=320)

        # Missing essentials
        missing = [n for n in ESSENTIALS if scores.get(n, 0) < 1]
        if missing:
            tips_list = [f"- **{n}**: {NUTRIENT_TIPS.get(n, '')}" for n in missing]
            st.warning("부족 태그:\n" + "\n".join(tips_list))
        else:
            st.success("핵심 태그 충족! (ESSENTIALS 기준)")

        # Intuitive one-liner
        try:
            st.info(make_intuitive_summary(scores, threshold=1.0))
        except Exception:
            pass

        # Per-meal breakdown (today-only)
        if source_mode == "오늘 기록 사용":
            df_today = today_food_log_df()
            df_meal = per_meal_breakdown(food_db, df_today)
            if not df_meal.empty:
                st.markdown("### 🍽️ 식사별 보충 포인트 (오늘)")
                for label in ["아침","점심","저녁","간식"]:
                    sub = df_meal[df_meal["시간대"] == label]
                    if sub.empty:
                        continue
                    st.markdown(f"**{label}**")
                    st.dataframe(sub[["시각","먹은 것","매칭","태그","한줄설명"]]
                                 .rename(columns={"시각":"시간"}),
                                 use_container_width=True,
                                 height=min(300, 60+28*len(sub)))
                    uniq = []
                    for s in sub["한줄설명"].tolist():
                        for part in [x.strip() for x in s.split("·")]:
                            if part and part not in uniq:
                                uniq.append(part)
                    if uniq:
                        st.caption("보충된 포인트: " + " · ".join(uniq[:6]))

        # Suggestions
        st.markdown("### 🍽️ 다음 식사 제안 (3가지)")
        # Diversity recent items (stub)
        recent_items = []
        try:
            if diversity_n > 0:
                r = ensure_log()
                r = r[r["type"] == "food"].copy()
                if not r.empty:
                    r["date"] = r["date"].astype(str)
                    r["time"] = r["time"].astype(str)
                    recent_df = r.sort_values(["date", "time"]).tail(diversity_n * 5)
                    recent_items = (recent_df["food_norm"].fillna("") + "|" + recent_df["item"].fillna("")).tolist()
                    recent_items = [x.split("|")[0] for x in recent_items if x]
        except Exception:
            recent_items = []

        rng = random.Random(hash(("quick-eval", text_in)) % (10**9))
        favor_tags = missing
        cols = st.columns(3)
        for idx in range(3):
            try:
                title, meal, explain = gen_meal(
                    food_db, include_caution,
                    recent_items=recent_items, favor_tags=favor_tags, rng=rng
                )
                with cols[idx]:
                    st.markdown(f"**{title} #{idx+1}**")
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
