#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diet_analyzer.py (selectable condition + persistent suggestions + non-toggle details)
"""

from __future__ import annotations
import re, sys, ast, json
from collections import defaultdict
from typing import List, Dict, Tuple, Any
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None

# ====================== 설정 ======================
FOOD_DB_CSV = "food_db.csv"
NUTRIENT_DICT_CSV = "nutrient_dict.csv"
LOG_CSV = "log.csv"

SLOTS = ["아침", "아침보조제", "오전 간식", "점심", "점심보조제",
         "오후 간식", "저녁", "저녁보조제", "저녁 간식"]

TZ = ZoneInfo("Europe/Paris")

# ==================== 날짜/상태 ====================
def today_str() -> str:
    return datetime.now(TZ).date().isoformat()

def init_daily_state():
    if "daily_date" not in st.session_state:
        st.session_state.daily_date = today_str()
    if st.session_state.daily_date != today_str():
        for k in ["inputs", "conditions", "last_items_df", "last_clicked_foods", "analyzed"]:
            st.session_state.pop(k, None)
        st.session_state.daily_date = today_str()

    st.session_state.setdefault("inputs", {s: "" for s in SLOTS})
    st.session_state.setdefault("conditions", {s: "" for s in SLOTS})
    st.session_state.setdefault("last_items_df", None)
    st.session_state.setdefault("last_clicked_foods", set())
    st.session_state.setdefault("analyzed", False)

# ==================== 유틸 ====================
def _parse_tags_from_slash(cell):
    if pd.isna(cell): return []
    return [t.strip() for t in str(cell).split('/') if t.strip()]

def _parse_taglist_cell(cell: Any):
    if isinstance(cell, list): return [str(x).strip() for x in cell if str(x).strip()]
    s = str(cell).strip() if cell else ""
    if not s or s == "[]": return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
    except: pass
    return [p.strip() for p in re.split(r"[,/]", s.strip("[]")) if p.strip()]

def load_food_db_simple(path=FOOD_DB_CSV):
    df = pd.read_csv(path)
    df["태그리스트"] = df.get("태그리스트", df.get("태그(영양)", "")).apply(_parse_tags_from_slash)
    return df[["식품", "등급", "태그리스트"]]

def load_nutrient_dict_simple(path=NUTRIENT_DICT_CSV):
    nd = pd.read_csv(path)
    return {str(r["영양소"]).strip(): str(r["한줄설명"]).strip() for _, r in nd.iterrows()}

# ================== 분석 함수 ==================
def split_items(text: str) -> List[str]:
    if not text: return []
    return [p.strip() for p in re.split(r"[,|\n|(|)]+", text) if p.strip()]

def match_item_to_foods(item, df_food):
    it = str(item).strip()
    hits = df_food[df_food["식품"].apply(lambda x: it in str(x) or str(x) in it)]
    return hits[hits["식품"].str.len() > 0]

def analyze_items_for_slot(input_text, slot, df_food, condition=""):
    items = split_items(input_text)
    per_item_rows, nutrient_counts = [], defaultdict(float)
    for raw in items:
        matched = match_item_to_foods(raw, df_food)
        if matched.empty:
            per_item_rows.append({"슬롯": slot, "입력항목": raw, "매칭식품": "", "태그": "", "컨디션": condition})
            continue
        tag_union, matched_names = [], []
        for _, r in matched.iterrows():
            name = str(r["식품"])
            matched_names.append(name)
            tags = r.get("태그리스트", [])
            for t in tags:
                tag_union.append(t)
                nutrient_counts[t] += 1
        per_item_rows.append({
            "슬롯": slot, "입력항목": raw,
            "매칭식품": ", ".join(matched_names),
            "태그": ", ".join(tag_union),
            "컨디션": condition
        })
    return pd.DataFrame(per_item_rows), dict(nutrient_counts)

# ================== 컨디션 → 태그 매핑 ==================
def condition_to_nutrients(condition: str) -> List[str]:
    cond = condition.lower()
    needs = []
    if any(k in cond for k in ["피곤", "무기력"]): needs += ["단백질", "비타민B", "철분"]
    if any(k in cond for k in ["복부팽만", "소화불량"]): needs += ["저FODMAP", "식이섬유(적당량)"]
    if "속쓰림" in cond: needs += ["저지방", "저산성"]
    if "두통" in cond or "어지럽" in cond: needs += ["마그네슘", "수분"]
    if "불면" in cond or "수면" in cond: needs += ["트립토판", "칼슘"]
    if "변비" in cond: needs += ["식이섬유", "수분"]
    if "설사" in cond: needs += ["전해질", "수분"]
    return list(dict.fromkeys(needs))

# ================== 태그 → 식품군 ==================
NUTRIENT_TO_FOODS = {
    "단백질": ["달걀", "닭가슴살", "두부", "그릭요거트", "생선"],
    "비타민B": ["현미", "통곡물빵", "콩류", "계란노른자"],
    "철분": ["시금치", "간", "붉은살생선", "렌틸콩"],
    "저FODMAP": ["호박", "당근", "감자", "쌀밥"],
    "식이섬유(적당량)": ["당근", "호박죽", "바나나"],
    "저지방": ["찐감자", "두부", "저지방요거트"],
    "저산성": ["바나나", "감자", "두유", "흰죽"],
    "마그네슘": ["견과류", "시금치", "카카오닙스"],
    "수분": ["국물", "과일", "물", "수프"],
    "트립토판": ["달걀", "귀리", "바나나", "아보카도"],
    "칼슘": ["요거트", "멸치", "치즈", "두유"],
    "전해질": ["바나나", "소금간 국물", "미음"]
}

# ================== 세부정보 표시 ==================
def show_food_details(food: str, df_food: pd.DataFrame, nutrient_desc: Dict[str, str]):
    matches = df_food[df_food["식품"].str.contains(food, case=False, na=False)]
    if matches.empty:
        st.warning(f"'{food}' 정보 없음")
        return
    with st.expander(f"🍽 {food} 세부정보 보기", expanded=True):
        for _, row in matches.iterrows():
            tags = row.get("태그리스트", [])
            st.write(f"**영양 태그:** {', '.join(tags) if tags else '없음'}")
            for t in tags:
                desc = nutrient_desc.get(t, "")
                if desc:
                    st.caption(f"• {t}: {desc}")

# ==================== Streamlit UI ====================
def main():
    st.set_page_config(page_title="식단 분석 및 제안", page_icon="🥗", layout="centered")
    st.title("🥗 슬롯별 식단 분석 · 다음 식사 제안")

    init_daily_state()
    df_food = load_food_db_simple()
    nutrient_desc = load_nutrient_dict_simple()

    condition_options = ["양호", "피곤함", "복부팽만", "속쓰림", "두통", "불면", "변비", "설사", "직접 입력"]

    for slot in SLOTS:
        val = st.text_area(slot, height=60, placeholder=f"{slot} 식단 입력", value=st.session_state.inputs.get(slot, ""))
        st.session_state.inputs[slot] = val

        prev_cond = st.session_state.conditions.get(slot, "")
        default_index = condition_options.index(prev_cond) if prev_cond in condition_options else len(condition_options) - 1
        selected = st.selectbox(f"{slot} 컨디션", condition_options, index=default_index, key=f"cond_select_{slot}")
        if selected == "직접 입력":
            cond_input = st.text_input(f"{slot} 컨디션 직접 입력", value=prev_cond if prev_cond not in condition_options else "")
            st.session_state.conditions[slot] = cond_input
        else:
            st.session_state.conditions[slot] = selected

    # -------------------------------
    # 분석하기 버튼 (상태 유지형)
    # -------------------------------
    if st.button("분석하기", type="primary"):
        st.session_state.analyzed = True
        st.session_state.last_clicked_foods.clear()

    if st.session_state.analyzed:
        all_items, total_counts = [], defaultdict(float)
        for slot in SLOTS:
            items_df, counts = analyze_items_for_slot(
                st.session_state.inputs.get(slot, ""), slot, df_food,
                st.session_state.conditions.get(slot, "")
            )
            all_items.append(items_df)
            for k, v in counts.items():
                total_counts[k] += v
        items_df_all = pd.concat(all_items, ignore_index=True) if all_items else pd.DataFrame()
        st.session_state.last_items_df = items_df_all

        # 🍽 제안 섹션
        st.markdown("### 🍽 개인화된 다음 식사 제안")
        total_tags = []
        if not items_df_all.empty and "태그" in items_df_all.columns:
            for tags in items_df_all["태그"].dropna():
                total_tags += [t.strip() for t in str(tags).split(",") if t.strip()]
        tag_counts = pd.Series(total_tags).value_counts().to_dict() if total_tags else {}

        for slot in SLOTS:
            cond = st.session_state.conditions.get(slot, "")
            if not cond.strip() or cond == "양호":
                continue
            needed_tags = condition_to_nutrients(cond)
            suggested_foods = []
            for tag in needed_tags:
                if tag_counts.get(tag, 0) < 1:
                    suggested_foods += NUTRIENT_TO_FOODS.get(tag, [])
            suggested_foods = list(dict.fromkeys(suggested_foods[:5]))

            if suggested_foods:
                st.markdown(f"#### 🩺 {slot} 컨디션: {cond}")
                cols = st.columns(len(suggested_foods))
                for i, food in enumerate(suggested_foods):
                    with cols[i]:
                        if st.button(food, key=f"suggest_btn_{slot}_{food}"):
                            st.session_state.last_clicked_foods.add(food)

        # 🔍 클릭된 식품 세부정보 표시
        if st.session_state.last_clicked_foods:
            st.markdown("### 🔍 선택한 식품 세부정보")
            for food in sorted(st.session_state.last_clicked_foods):
                show_food_details(food, df_food, nutrient_desc)

    # 결과표
    st.markdown("### 🍱 슬롯별 매칭 결과")
    if st.session_state.last_items_df is not None and not st.session_state.last_items_df.empty:
        st.dataframe(st.session_state.last_items_df, use_container_width=True)
    else:
        st.info("아직 분석 결과가 없습니다.")

if __name__ == "__main__":
    if st is None:
        print("Streamlit is required. Run with: pip install streamlit")
        sys.exit(1)
    main()