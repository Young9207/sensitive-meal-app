#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diet_analyzer_v2.py
- 각 식사별 입력 + 컨디션 기록
- 영양소별 태그 집계
- 부족 영양소 기반 다음 식사 제안 (자동 threshold 보정)
- log.csv / food_db_updated.csv 자동 관리
"""

from __future__ import annotations
import re, sys, ast, json, base64, zlib
from collections import defaultdict
from typing import List, Dict, Tuple, Any
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None

# ================= 설정 =================
FOOD_DB_CSV = "food_db.csv"
NUTRIENT_DICT_CSV = "nutrient_dict.csv"
LOG_CSV = "log.csv"
FOOD_DB_UPDATED_CSV = "food_db_updated.csv"
SLOTS = ["아침", "오전 간식", "점심", "오후 간식", "저녁"]
TZ = ZoneInfo("Europe/Paris")


# ================= 날짜 관리 =================
def today_str() -> str:
    return datetime.now(TZ).date().isoformat()


def next_midnight():
    now = datetime.now(TZ)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=TZ)


def init_state():
    """자정 단위 초기화"""
    if "daily_date" not in st.session_state:
        st.session_state.daily_date = today_str()
    if st.session_state.daily_date != today_str():
        for k in ["inputs", "conditions", "last_items", "nutri_summary", "recs"]:
            st.session_state.pop(k, None)
        st.session_state.daily_date = today_str()
    st.session_state.setdefault("inputs", {s: "" for s in SLOTS})
    st.session_state.setdefault("conditions", {s: "" for s in SLOTS})
    st.session_state.setdefault("last_items", pd.DataFrame())
    st.session_state.setdefault("nutri_summary", pd.DataFrame())
    st.session_state.setdefault("recs", [])
    st.session_state.setdefault("threshold", 1)


# ================= 데이터 로딩 =================
def _parse_taglist_cell(cell) -> list[str]:
    if isinstance(cell, list):
        return cell
    if pd.isna(cell) or str(cell).strip() in ("", "[]"):
        return []
    s = str(cell).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v]
    except Exception:
        pass
    s2 = s.strip("[]")
    return [p.strip().strip("'").strip('"') for p in re.split(r"[,/]", s2) if p.strip()]


def load_food_db(path=FOOD_DB_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "태그리스트" not in df.columns:
        df["태그리스트"] = df["태그(영양)"].apply(lambda x: _parse_taglist_cell(x))
    else:
        df["태그리스트"] = df["태그리스트"].apply(_parse_taglist_cell)
    return df


def load_nutrient_dict(path=NUTRIENT_DICT_CSV) -> dict:
    nd = pd.read_csv(path)
    return {str(r["영양소"]).strip(): str(r["한줄설명"]).strip() for _, r in nd.iterrows()}


# ================= 파서 =================
def split_items(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[,|\n|(|)|+]+", text)
    return [p.strip() for p in parts if p.strip()]


def parse_qty(token: str) -> tuple[str, float]:
    m = re.search(r"(.*?)(\d+(?:\.\d+)?)$", token)
    return (m.group(1).strip(), float(m.group(2))) if m else (token.strip(), 1.0)


# ================= 분석 =================
def match_item_to_foods(item, df):
    item_norm = item.strip()
    hits = df[df["식품"].apply(lambda x: item_norm in str(x) or str(x) in item_norm)]
    return hits


def analyze_items(inputs: dict, df_food, nutrient_desc):
    total_counts = defaultdict(float)
    rows, logs, unmatched = [], [], []

    for slot, text in inputs.items():
        tokens = [parse_qty(t) for t in split_items(text)]
        for name, qty in tokens:
            hits = match_item_to_foods(name, df_food)
            timestamp = datetime.now(TZ).isoformat(timespec="seconds")
            if hits.empty:
                unmatched.append(name)
                logs.append([timestamp, today_str(), slot, name, qty, "", "", ""])
                continue

            tag_union, matched_names = [], []
            agg_grade = "Safe"
            for _, r in hits.iterrows():
                grade = r.get("등급", "Safe")
                tags = _parse_taglist_cell(r["태그리스트"])
                matched_names.append(r["식품"])
                for t in tags:
                    total_counts[t] += qty
                    tag_union.append(t)
            rows.append([slot, name, qty, ", ".join(matched_names), agg_grade, ", ".join(set(tag_union))])
            logs.append([timestamp, today_str(), slot, name, qty, ", ".join(matched_names), agg_grade, ", ".join(set(tag_union))])
    return pd.DataFrame(rows, columns=["슬롯","입력항목","수량","매칭식품","등급","태그"]), total_counts, unmatched, logs


# ================= 다음 식사 제안 =================
def recommend_next_meal(total_counts, df_food, nutrient_desc, top_nutrients=3):
    """상대적 부족 영양소 기반 추천"""
    if not total_counts:
        return [{"부족영양소": "데이터 없음", "설명": "식사 기록이 없습니다.", "추천식품": []}]

    avg = sum(total_counts.values()) / (len(total_counts) or 1)
    deficits = {t: max(0.1, avg * 0.8 - v) for t, v in total_counts.items() if v < avg * 0.8}
    if not deficits:
        return [{"부족영양소": "균형 유지", "설명": "영양 균형이 양호합니다.", "추천식품": []}]

    lacking_sorted = sorted(deficits.items(), key=lambda x: x[1], reverse=True)[:top_nutrients]
    focus_tags = [t for t, _ in lacking_sorted]

    recs = []
    for tag in focus_tags:
        foods = df_food[df_food["태그리스트"].apply(lambda lst: tag in lst and "Safe" in df_food["등급"].values)]
        foods = foods["식품"].head(5).tolist()
        recs.append({"부족영양소": tag, "설명": nutrient_desc.get(tag, ""), "추천식품": foods})
    return recs


# ================= Streamlit UI =================
def main():
    st.set_page_config(page_title="다음 식사 제안", page_icon="🥗")
    st.title("🥗 식단 분석 + 다음 식사 제안")

    init_state()

    df_food = load_food_db()
    nutrient_desc = load_nutrient_dict()

    d = st.date_input("기록 날짜", value=date.today())
    st.caption("쉼표(,)로 구분해 입력하세요. 예: 닭가슴살, 현미밥, 김치")

    for slot in SLOTS:
        st.session_state.inputs[slot] = st.text_area(slot, value=st.session_state.inputs.get(slot, ""), height=60)
        st.session_state.conditions[slot] = st.text_input(f"{slot} 컨디션", value=st.session_state.conditions.get(slot, ""), placeholder="예: 피곤함, 복부팽만")

    if st.button("분석하기", type="primary"):
        items_df, counts, unmatched, logs = analyze_items(st.session_state.inputs, df_food, nutrient_desc)
        st.session_state.last_items = items_df

        # 영양 요약
        summary = pd.DataFrame([{"영양소": k, "수량합": v} for k, v in counts.items()]).sort_values("수량합", ascending=False)
        st.session_state.nutri_summary = summary

        # 다음 식사 제안
        recs = recommend_next_meal(counts, df_food, nutrient_desc)
        st.session_state.recs = recs

        # 로그 저장
        log_df = pd.DataFrame(logs, columns=["timestamp","date","slot","입력항목","수량","매칭식품","등급","태그"])
        try:
            prev = pd.read_csv(LOG_CSV)
            merged = pd.concat([prev, log_df], ignore_index=True)
        except Exception:
            merged = log_df
        merged.to_csv(LOG_CSV, index=False, encoding="utf-8-sig")

        # 미매칭 식품 추가
        if unmatched:
            new_rows = [{"식품": n, "등급": "", "태그(영양)": "", "태그리스트": []} for n in unmatched]
            updated = pd.concat([df_food, pd.DataFrame(new_rows)], ignore_index=True)
            updated.to_csv(FOOD_DB_UPDATED_CSV, index=False, encoding="utf-8-sig")

    # 결과 표시
    st.markdown("### 🍱 슬롯별 매칭 결과")
    if not st.session_state.last_items.empty:
        st.dataframe(st.session_state.last_items, use_container_width=True)
    else:
        st.info("입력된 식사 데이터가 없습니다.")

    st.markdown("### 🧭 영양 태그 요약")
    if not st.session_state.nutri_summary.empty:
        st.dataframe(st.session_state.nutri_summary, use_container_width=True)
    else:
        st.info("아직 분석 결과가 없습니다.")

    st.markdown("### 🍽 다음 식사 제안")
    if st.session_state.recs:
        for r in st.session_state.recs:
            foods = ", ".join(r["추천식품"]) if r["추천식품"] else "(추천 식품 없음)"
            st.write(f"- **{r['부족영양소']}** → {r['설명']}")
            st.caption(f"추천 식품: {foods}")
    else:
        st.info("분석 후 제안이 여기에 표시됩니다.")


if __name__ == "__main__":
    if st is None:
        print("Streamlit 설치 필요: pip install streamlit")
        sys.exit(1)
    main()