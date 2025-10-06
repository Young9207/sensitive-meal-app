# -*- coding: utf-8 -*-
"""
Instant Diet Evaluator (즉석 식단 평가) — with Nutrient Tips
-----------------------------------------------------------
Paste your meal list, get tag scores, and see 1-line nutrient tips.
Also suggests the next meal (3 options) focusing on missing essentials.

Run:
    streamlit run instant_diet_eval_app.py
"""
import re
import random
from difflib import get_close_matches
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st

# -----------------------------
# Minimal in-app "DB"
# -----------------------------
CORE_NUTRIENTS = [
    "단백질", "식이섬유", "철", "칼슘", "마그네슘", "칼륨",
    "오메가3", "비타민A", "비타민B", "비타민C", "비타민D", "비타민E",
    "저당", "저염", "건강한지방"
]

# Essentials considered "must-have per day" (toy rule)
ESSENTIALS = ["단백질", "식이섬유", "비타민C", "칼슘"]

# NEW: 1-line tips per nutrient tag (plain-language, practical)
NUTRIENT_TIPS: Dict[str, str] = {
    "단백질": "근육 유지와 포만감에 좋아요—식사 후 허기를 줄여줘요.",
    "식이섬유": "배변 리듬과 포만감에 도움—당 흡수도 완만하게 해줘요.",
    "철": "피로감 줄이고 집중에 도움—혈액 산소 운반에 필수예요.",
    "칼슘": "뼈·치아 건강에 기본—근육 수축에도 필요해요.",
    "마그네슘": "긴장 완화와 수면·근육 기능에 도움을 줘요.",
    "칼륨": "나트륨 배출을 도와 붓기·혈압 관리에 좋아요.",
    "오메가3": "심혈관 건강과 염증 균형에 도움—기름진 생선에 많아요.",
    "비타민A": "눈·피부 점막 보호—색 진한 채소에 풍부해요.",
    "비타민B": "에너지 대사 서포트—피로감 낮추는 데 도움.",
    "비타민C": "면역과 철 흡수 UP—가열 덜 한 채소·과일로.",
    "비타민D": "뼈 건강과 면역에 도움—햇빛·달걀·생선에도 있어요.",
    "비타민E": "세포 보호(항산화)와 피부 컨디션에 도움.",
    "저당": "식후 혈당 출렁임을 줄이는 선택이에요.",
    "저염": "붓기·혈압 관리에 유리—가공식품 소금 확인!",
    "건강한지방": "포만감·지용성 비타민 흡수 도우미(아보카도·견과).",
    # Extra tags that may appear in sample DB
    "저지방": "열량 대비 단백질 채우기 좋고 소화도 비교적 편해요."
}

# Basic food DB for demo
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

VIRTUAL_RULES: Dict[str, Dict[str, Any]] = {}
DEFAULT_USER_RULES = {"avoid_keywords": [], "allow_keywords": []}

# -----------------------------
# Helpers
# -----------------------------
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
    text = text.lower()
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
    return pd.DataFrame(columns=["type", "date", "time", "food_norm", "item"])

def pick_candidates_by_tags(df: pd.DataFrame, favor_tags: List[str]) -> List[str]:
    if not favor_tags:
        return df["식품"].tolist()
    rows = []
    for _, r in df.iterrows():
        tags = r.get("태그(영양)", [])
        overlap = len(set(tags) & set(favor_tags))
        rows.append((overlap, r["식품"]))
    rows.sort(key=lambda x: (-x[0], x[1]))
    return [name for score, name in rows if score > 0] or df["식품"].tolist()

def gen_meal(
    df: pd.DataFrame,
    include_caution: bool,
    mode: str = "기본",
    recent_items: List[str] = None,
    favor_tags: List[str] = None,
    rng: random.Random = None,
    user_rules: Dict[str, Any] = None,
    allow_rare: bool = False
) -> Tuple[str, List[str], str]:
    rng = rng or random.Random()
    recent_items = set(recent_items or [])
    favor_tags = favor_tags or []

    df2 = df.copy()
    if not include_caution:
        df2 = df2[df2["등급"] != "Caution"]

    if user_rules and user_rules.get("avoid_keywords"):
        mask = df2["식품"].apply(lambda x: not contains_any(x, user_rules["avoid_keywords"]))
        df2 = df2[mask]

    pool = pick_candidates_by_tags(df2, favor_tags)
    pool = [p for p in pool if p not in recent_items]
    if len(pool) < 3:
        pool = df2["식품"].tolist()

    picks = rng.sample(pool, k=min(3, len(pool))) if len(pool) >= 3 else pool
    explain = ""
    if favor_tags:
        explain = f"부족 태그 보완 중심: {', '.join(favor_tags)}"
    title = "다음 식사 제안"
    return title, picks, explain

# -----------------------------
# Core scoring
# -----------------------------
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
            if mapped in VIRTUAL_RULES:
                vr = VIRTUAL_RULES[mapped]
                grade, flags, tags = vr.get("grade", "Safe"), vr.get("flags", ""), vr.get("tags", [])
                if contains_any(name_norm, user_rules.get("allow_keywords", [])):
                    grade, flags = "Safe", "개인 허용"
            else:
                rec = df_food[df_food["식품"] == mapped].iloc[0]
                grade = rec.get("등급", "Safe")
                tags = rec.get("태그(영양)", [])
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
            if df_new["태그(영양)"].dtype == object:
                def parse_tags(x):
                    if isinstance(x, list):
                        return x
                    parts = re.split(r"[\/,]+", str(x))
                    return [p.strip() for p in parts if p.strip()]
                df_new["태그(영양)"] = df_new["태그(영양)"].apply(parse_tags)
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
sample = "쌀밥1, 대구구이1, 양배추1, 당근1, 올리브유0.5"
text_in = st.text_area("식단 텍스트 (쉼표 또는 줄바꿈으로 구분, 예: "+sample+")", height=120, placeholder=sample)

col_btn1, col_btn2 = st.columns([1,1])
with col_btn1:
    analyze = st.button("분석하기", type="primary")
with col_btn2:
    clear = st.button("지우기")
if clear:
    text_in = ""
    st.experimental_rerun()

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
        st.dataframe(score_df, use_container_width=True, height=360)

        # Missing essentials
        missing = [n for n in ESSENTIALS if scores.get(n, 0) < 1]
        if missing:
            tips_list = [f"- **{n}**: {NUTRIENT_TIPS.get(n, '')}" for n in missing]
            st.warning("부족 태그:\n" + "\n".join(tips_list))
        else:
            st.success("핵심 태그 충족! (ESSENTIALS 기준)")

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

        st.markdown("### 🍽️ 다음 식사 제안 (3가지)")
        rng = random.Random(hash(("quick-eval", text_in)) % (10**9))
        favor_tags = missing  # focus on missing
        cols = st.columns(3)
        for idx in range(3):
            try:
                title, meal, explain = gen_meal(
                    food_db,
                    include_caution,
                    mode="기본",
                    recent_items=recent_items,
                    favor_tags=favor_tags,
                    rng=rng,
                    user_rules=user_rules,
                    allow_rare=False
                )
                with cols[idx]:
                    st.markdown(f"**{title} #{idx+1}**")
                    st.write(" / ".join(meal))
                    if favor_tags:
                        # show short reasons based on tips (first 2 tags to keep it short)
                        why = [f"· {t}: {NUTRIENT_TIPS.get(t, '')}" for t in favor_tags[:2]]
                        st.caption("보완 포인트:\n" + "\n".join(why))
                    elif explain:
                        st.caption(explain)
            except Exception as e:
                st.error(f"제안 생성 실패: {e}")
    except Exception as e:
        st.error(f"분석 실패: {e}")
