#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diet_analyzer.py
입력(아침/오전 간식/점심/오후 간식/저녁) → 식품 매칭 → 영양 분석 → 다음 식사 제안 → log.csv 저장
+ 매칭 안된 재료는 food_db에 신규 식품 행으로 추가하여 food_db_updated.csv 생성

- 필요 파일: food_db.csv (식품, 등급, 태그(영양), 태그리스트), nutrient_dict.csv(영양소, 한줄설명 ...)
- 실행: streamlit run diet_analyzer.py
- 파싱 규칙:
  * 쉼표(,) / 줄바꿈으로 1차 분리 → 각 토큰을 '+' 로 2차 분리
  * 토큰 끝 숫자는 수량으로 해석 (예: '우메보시2' → 이름='우메보시', 수량=2.0) 없으면 1.0
"""

import re
import sys
import ast
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Any
from datetime import datetime, date

import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None  # allow import without Streamlit


# ====================== 설정 ======================
FOOD_DB_CSV = "food_db.csv"
NUTRIENT_DICT_CSV = "nutrient_dict.csv"
LOG_CSV = "log.csv"
FOOD_DB_UPDATED_CSV = "food_db_updated.csv"

SLOTS = ["아침", "오전 간식", "점심", "오후 간식", "저녁"]


# ==================== 유틸/전처리 ====================
def _parse_tags_from_slash(cell) -> List[str]:
    if pd.isna(cell):
        return []
    return [t.strip() for t in str(cell).split('/') if t.strip()]


def _parse_taglist_cell(cell: Any) -> List[str]:
    """
    태그리스트 셀을 '항상 리스트'로 변환.
    허용 포맷:
      - 파이썬 리스트 문자열: "['단백질', '저지방']"
      - JSON 배열 문자열: '["단백질","저지방"]'
      - 슬래시 구분: "단백질/저지방"
      - 실제 리스트: ['단백질', '저지방']
      - 빈 값/[] → []
    """
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    s = "" if cell is None or (isinstance(cell, float) and pd.isna(cell)) else str(cell).strip()
    if not s or s == "[]":
        return []
    # 1) literal_eval 시도 (파이썬 리스트 표기)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    # 2) JSON 파싱 시도
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    # 3) 슬래시/쉼표 구분 문자열로 처리
    #    (따옴표/대괄호 흔적 제거)
    s2 = s.strip().strip("[]")
    parts = [p.strip().strip("'").strip('"') for p in re.split(r"[,/]", s2) if p.strip()]
    return [p for p in parts if p]


def _ensure_taglist(lst_from_row: Any, fallback_slash: Any) -> List[str]:
    """
    우선 태그리스트 파싱 → 비어있으면 태그(영양) 슬래시 분리로 대체
    """
    tags = _parse_taglist_cell(lst_from_row)
    if not tags:
        tags = _parse_tags_from_slash(fallback_slash)
    return tags


def load_food_db_simple(path: str = FOOD_DB_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 필수 컬럼 보장
    for c in ["식품", "등급", "태그(영양)"]:
        if c not in df.columns:
            df[c] = ""

    # 태그리스트 정규화 (있든 없든 항상 리스트로)
    if "태그리스트" in df.columns:
        df["태그리스트"] = df["태그리스트"].apply(_parse_taglist_cell)
    else:
        df["태그리스트"] = df["태그(영양)"].apply(_parse_tags_from_slash)

    return df[["식품", "등급", "태그(영양)", "태그리스트"]]



def load_nutrient_dict_simple(path: str = NUTRIENT_DICT_CSV) -> Dict[str, str]:
    nd = pd.read_csv(path)
    for c in ["영양소", "한줄설명"]:
        if c not in nd.columns:
            nd[c] = ""
    return {str(r["영양소"]).strip(): str(r["한줄설명"]).strip() for _, r in nd.iterrows()}


_GRADE_ORDER = {"Avoid": 2, "Caution": 1, "Safe": 0}


def _worse_grade(g1: str, g2: str) -> str:
    return g1 if _GRADE_ORDER.get(g1, 0) >= _GRADE_ORDER.get(g2, 0) else g2


def _norm(s: str) -> str:
    return str(s or "").strip()

#=================time
from datetime import datetime, date, timedelta, timezone

KST = timezone(timedelta(hours=9))  # 타임존 쓰시면 맞춰서
def today_str():
    return datetime.now(KST).date().isoformat()

def next_midnight():
    now = datetime.now(KST)
    nm = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return nm

def init_daily_state():
    """자정 단위로 state를 유지. 날짜 바뀌면 자동 초기화."""
    # 날짜 키
    if "daily_date" not in st.session_state:
        st.session_state.daily_date = today_str()

    # 날짜가 바뀌었으면 초기화
    if st.session_state.daily_date != today_str():
        # 초기화할 키들
        for k in ["inputs", "last_items_df", "last_nutri_df", "last_recs", "last_combo"]:
            st.session_state.pop(k, None)
        st.session_state.daily_date = today_str()

    # 슬롯 입력 저장소
    if "inputs" not in st.session_state:
        st.session_state.inputs = {s: "" for s in SLOTS}

    # 결과 저장소
    st.session_state.setdefault("last_items_df", None)
    st.session_state.setdefault("last_nutri_df", None)
    st.session_state.setdefault("last_recs", [])
    st.session_state.setdefault("last_combo", [])

# ==================== 파서 (콤마/플러스/수량) ====================
def split_items(text: str) -> List[str]:
    """쉼표(,)와 줄바꿈으로 먼저 분리한 뒤, 각 토큰을 '+'로 추가 분리"""
    if not text:
        return []
    first = [p.strip() for p in re.split(r"[,|\n|(|)]+", text) if p.strip()]
    final = []
    for part in first:
        final += [q.strip() for q in part.split('+') if q.strip()]
    return final


def parse_qty(token: str) -> Tuple[str, float]:
    """토큰 끝의 숫자를 수량으로 파싱. 없으면 1.0"""
    m = re.search(r"(.*?)(\d+(?:\.\d+)?)\s*$", token)
    if m:
        name = m.group(1).strip()
        qty = float(m.group(2))
        return name, qty
    return token.strip(), 1.0


# ================== 매칭/분석/추천 로직 ==================
def match_item_to_foods(item: str, df_food: pd.DataFrame) -> pd.DataFrame:
    """item(예: '소고기 미역국')에 대해 food_db의 식품명이 포함되면 매칭.
       반대방향(항목명이 더 짧고 DB가 길 때)도 허용."""
    it = _norm(item)
    hits = df_food[
        df_food["식품"].apply(lambda x: _norm(x) in it or it in _norm(x))
    ].copy()
    hits = hits[hits["식품"].apply(lambda x: len(_norm(x)) >= 1)]
    return hits


def analyze_items_for_slot(input_text: str, slot: str, df_food: pd.DataFrame, nutrient_desc: Dict[str, str]):
    """슬롯 단위 분석 → (items_df, nutrient_counts(dict), log_df, unmatched_names(list))"""
    raw_tokens = split_items(input_text)
    items = [parse_qty(tok) for tok in raw_tokens]  # [(name, qty), ...]

    per_item_rows = []
    nutrient_counts = defaultdict(float)
    log_rows = []
    unmatched_names = []

    for raw, qty in items:
        if not raw:
            continue
        matched = match_item_to_foods(raw, df_food)
        timestamp = datetime.now().isoformat(timespec="seconds")
        if matched.empty:
            per_item_rows.append({
                "슬롯": slot, "입력항목": raw, "수량": qty, "매칭식품": "", "등급": "", "태그": ""
            })
            log_rows.append({
                "timestamp": timestamp,
                "date": date.today().isoformat(),
                "time": timestamp.split("T")[1],
                "slot": slot,
                "입력항목": raw, "수량": qty, "매칭식품": "", "등급": "", "태그": ""
            })
            unmatched_names.append(_norm(raw))
            continue

        agg_grade = "Safe"
        tag_union = []
        matched_names = []

        for _, r in matched.iterrows():
            name = _norm(r["식품"])
            grade = _norm(r["등급"]) or "Safe"
            tags = r.get("태그리스트", [])
            # 안전장치: 혹시라도 문자열이면 파싱
            if not isinstance(tags, list):
                tags = _parse_taglist_cell(tags)
            if not tags:
                tags = _parse_tags_from_slash(r.get("태그(영양)", ""))

            agg_grade = _worse_grade(agg_grade, grade)
            matched_names.append(name)
            for t in tags:
                if t:
                    tag_union.append(t)
                    nutrient_counts[t] += float(qty or 1.0)  # 수량 반영

        per_item_rows.append({
            "슬롯": slot,
            "입력항목": raw,
            "수량": qty,
            "매칭식품": ", ".join(dict.fromkeys(matched_names)),
            "등급": agg_grade,
            "태그": ", ".join(dict.fromkeys(tag_union))
        })
        log_rows.append({
            "timestamp": timestamp,
            "date": date.today().isoformat(),
            "time": timestamp.split("T")[1],
            "slot": slot,
            "입력항목": raw,
            "수량": qty,
            "매칭식품": ", ".join(dict.fromkeys(matched_names)),
            "등급": agg_grade,
            "태그": ", ".join(dict.fromkeys(tag_union))
        })

    return (
        pd.DataFrame(per_item_rows),
        dict(nutrient_counts),
        pd.DataFrame(log_rows),
        unmatched_names
    )


def summarize_nutrients(
    nutrient_counts: Dict[str, float],
    df_food: pd.DataFrame,
    nutrient_desc: Dict[str, str],
    threshold: int = 1
) -> pd.DataFrame:
    # 태그 우주
    all_tags = sorted({
        t for tlist in df_food["태그리스트"].apply(_parse_taglist_cell) for t in tlist
    })

    # 태그가 하나도 없으면 '빈 테이블이지만 컬럼은 있는' DataFrame 반환
    if not all_tags:
        return pd.DataFrame(columns=["영양소", "수량합", "상태", "한줄설명"])

    rows = []
    for tag in all_tags:
        cnt = float(nutrient_counts.get(tag, 0))
        rows.append({
            "영양소": tag,
            "수량합": cnt,
            "상태": "충족" if cnt >= threshold else "부족",
            "한줄설명": nutrient_desc.get(tag, "")
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["상태", "수량합", "영양소"], ascending=[True, False, True])


# def recommend_next_meal(nutrient_counts: Dict[str, float], df_food: pd.DataFrame, nutrient_desc: Dict[str, str],
#                         top_nutrients: int = 2, per_food: int = 4):
#     """부족 영양소 중심 추천: Safe 식품 우선 + 간단 조합"""
#     # 태그 우주 생성 시에도 안정적으로 리스트 파싱
#     tag_universe = {tt for lst in df_food["태그리스트"].apply(_parse_taglist_cell) for tt in lst}
#     tag_totals = {t: float(nutrient_counts.get(t, 0)) for t in tag_universe}
#     lacking = [t for t, v in sorted(tag_totals.items(), key=lambda x: x[1]) if v < 1.0]
#     lacking = lacking[:top_nutrients]

#     suggestions = []
#     for tag in lacking:
#         pool = df_food[
#             (df_food["등급"] == "Safe") &
#             (df_food["태그리스트"].apply(lambda lst: tag in _parse_taglist_cell(lst)))
#         ]
#         foods = pool["식품"].dropna().astype(str).head(per_food).tolist()
#         suggestions.append({
#             "부족영양소": tag,
#             "설명": nutrient_desc.get(tag, ""),
#             "추천식품": foods
#         })

#     combo = []
#     for s in suggestions:
#         for f in s["추천식품"]:
#             if f not in combo:
#                 combo.append(f)
#             if len(combo) >= 4:
#                 break
#         if len(combo) >= 4:
#             break

#     return suggestions, combo

# ========= 개선된 다음 식사 제안 =========
def _tag_deficits(nutrient_counts: Dict[str, float],
                  tag_universe: List[str],
                  tag_targets: Dict[str, float] | None = None) -> Dict[str, float]:
    """태그별 부족량(>0) 계산"""
    tag_targets = tag_targets or {}
    deficits = {}
    for t in tag_universe:
        target = float(tag_targets.get(t, 1.0))  # 기본 목표 = 1
        cur = float(nutrient_counts.get(t, 0.0))
        lack = max(0.0, target - cur)
        if lack > 0:
            deficits[t] = lack
    return deficits


def _food_score(tags: list[str],
                deficits: Dict[str, float],
                grade: str,
                prefer_tags: set[str],
                avoid_tags: set[str],
                grade_weights: dict[str, float]) -> float:
    """후보 식품의 점수: 부족 채움 가중치 × 등급 가중치 - 패널티"""
    if not tags:
        return 0.0

    # 채워주는 양(부족 태그 합)
    gain = sum(deficits.get(t, 0.0) for t in tags)

    # 등급 가중치 (Safe=1.0, Caution=0.6, Avoid=0)
    gw = grade_weights.get(grade, 0.0)
    score = gain * gw

    # 선호/회피 태그 가중(작게)
    if prefer_tags:
        score += 0.1 * sum(1.0 for t in tags if t in prefer_tags)
    if avoid_tags:
        score -= 0.2 * sum(1.0 for t in tags if t in avoid_tags)

    return score


def recommend_next_meal(nutrient_counts: Dict[str, float],
                        df_food: pd.DataFrame,
                        nutrient_desc: Dict[str, str],
                        *,
                        # 옵션 파라미터(필요 시 바꾸기)
                        tag_targets: Dict[str, float] | None = None,   # 태그별 목표치 (기본 1.0)
                        prefer_tags: list[str] | None = None,         # 선호 태그 (예: ['단백질','식이섬유'])
                        avoid_tags: list[str] | None = None,          # 회피 태그 (예: ['당','탄수화물'])
                        allowed_grades: tuple[str, ...] = ('Safe','Caution'),  # 제안 허용 등급
                        grade_weights: dict[str, float] = None,       # 등급 가중치
                        top_nutrients: int = 3,       # 상위 부족 태그 n개만 집중
                        per_food: int = 6,            # 태그별 최대 후보 노출
                        max_items: int = 4            # 제안 묶음 길이(최대 몇 가지 조합?)
                        ):
    """
    개선 포인트
      - 태그별 목표치 대비 '부족량'을 계산하고 그 부족을 가장 잘 메우는 식품을 고름
      - Safe 우선, Caution은 패널티, Avoid는 기본적으로 제외
      - 선호/회피 태그 반영(소폭 가중)
      - 그리디로 서로 다른 태그를 최대한 커버하는 1~max_items 조합 생성
    반환: (부족태그별 추천 테이블, 제안 조합 리스트)
    """
    prefer_tags = set(prefer_tags or [])
    avoid_tags = set(avoid_tags or [])
    if grade_weights is None:
        grade_weights = {'Safe': 1.0, 'Caution': 0.6, 'Avoid': 0.0}

    # 태그 우주 & 부족량
    tag_universe = sorted({t for lst in df_food["태그리스트"].apply(_parse_taglist_cell) for t in lst})
    if not tag_universe:
        return [], []

    deficits_all = _tag_deficits(nutrient_counts, tag_universe, tag_targets)
    if not deficits_all:
        return [], []  # 이미 목표 충족

    # 상위 부족 태그만 집중
    lacking_sorted = sorted(deficits_all.items(), key=lambda x: x[1], reverse=True)
    focus_tags = {t for t, _ in lacking_sorted[:max(1, top_nutrients)]}

    # 후보 풀: 허용 등급만, 태그가 하나라도 focus 태그에 걸리는 식품
    cand = df_food[df_food["등급"].isin(allowed_grades)].copy()
    cand["태그리스트"] = cand["태그리스트"].apply(_parse_taglist_cell)
    cand = cand[cand["태그리스트"].apply(lambda lst: any(t in focus_tags for t in lst))]

    if cand.empty:
        return [], []

    # 식품별 점수 계산
    cand["__score"] = cand.apply(
        lambda r: _food_score(
            r["태그리스트"],
            deficits_all,
            str(r["등급"]),
            prefer_tags,
            avoid_tags,
            grade_weights
        ), axis=1
    )

    cand = cand.sort_values("__score", ascending=False)

    # 부족 태그별 top 후보 뽑아 설명용 테이블 구성
    suggestions = []
    for tag in sorted(focus_tags, key=lambda t: deficits_all[t], reverse=True):
        pool = cand[cand["태그리스트"].apply(lambda lst: tag in lst)]
        foods = pool["식품"].head(per_food).tolist()
        suggestions.append({
            "부족영양소": tag,
            "설명": nutrient_desc.get(tag, ""),
            "추천식품": foods
        })

    # 그리디 조합: 점수 높은 순으로 뽑되, '아직 부족한 태그'를 더 많이 채우는 아이를 우선
    remaining = deficits_all.copy()
    picked = []
    for _, row in cand.iterrows():
        if len(picked) >= max_items:
            break
        tags = row["태그리스트"]
        # 이 식품이 남아있는 부족을 실질적으로 줄이는가?
        gain = sum(remaining.get(t, 0.0) for t in tags)
        if gain <= 0:
            continue
        picked.append(str(row["식품"]))
        # 남은 부족 업데이트 (한 번 선택 시 해당 태그 부족을 최대 1.0만큼만 줄인다고 가정)
        for t in tags:
            if t in remaining:
                remaining[t] = max(0.0, remaining[t] - 1.0)

    return suggestions, picked[:max_items]


def append_unmatched_to_food_db(df_food: pd.DataFrame, unmatched_names: List[str]) -> pd.DataFrame:
    """매칭 안 된 식품명을 식품 컬럼으로 추가(등급/태그 비워둠). 이미 존재하면 건너뜀."""
    to_add = []
    existing = set(df_food["식품"].astype(str).str.strip().tolist())
    for name in unmatched_names:
        if not name:
            continue
        if name in existing:
            continue
        to_add.append({"식품": name, "등급": "", "태그(영양)": "", "태그리스트": []})
        existing.add(name)
    if to_add:
        df_new = pd.concat([df_food, pd.DataFrame(to_add)], ignore_index=True)
    else:
        df_new = df_food.copy()
    return df_new


# ==================== Streamlit UI ====================
def main():
    # 페이지 설정 & 제목
    st.set_page_config(page_title="슬롯별 식단 분석 · 다음 식사 제안", page_icon="🥗", layout="centered")
    st.title("🥗 슬롯별 식단 분석 · 다음 식사 제안")

    # ✅ 자정 기준 하루 메모리 초기화 (init_daily_state, next_midnight, KST는 상단 유틸에 정의)
    init_daily_state()
    remain = (next_midnight() - datetime.now(KST))
    st.caption(f"현재 입력/결과는 **자정까지 자동 보존**됩니다. 남은 시간: 약 {remain.seconds//3600}시간 {remain.seconds%3600//60}분")

    # 파일 로딩
    with st.expander("데이터 파일 경로 설정", expanded=False):
        food_path = st.text_input("food_db.csv 경로", value=FOOD_DB_CSV)
        nutri_path = st.text_input("nutrient_dict.csv 경로", value=NUTRIENT_DICT_CSV)
        load_btn = st.button("파일 다시 로드")
    if load_btn:
        st.experimental_rerun()

    # 실제 로딩
    try:
        df_food = load_food_db_simple(food_path if 'food_path' in locals() else FOOD_DB_CSV)
        nutrient_desc = load_nutrient_dict_simple(nutri_path if 'nutri_path' in locals() else NUTRIENT_DICT_CSV)
    except Exception as e:
        st.error(f"CSV 로딩 중 오류: {e}")
        st.stop()

    st.caption("입력 예: 소고기 미역국, 찹쌀밥, 총각김치1, 무쌈, 우메보시2, 닭고기2, 들기름, 올리브유 사과+시나몬가루, 블랙커피1")

    # 날짜(기록용)
    d = st.date_input("기록 날짜", value=date.today())

    # 슬롯별 입력 (session_state에 보존)
    with st.container():
     for slot in SLOTS:
         val = st.text_area(
             slot, height=70, placeholder=f"{slot}에 먹은 것 입력",
             key=f"ta_{slot}",
             value=st.session_state.inputs.get(slot, "")
         )
         st.session_state.inputs[slot] = val  # <- 이 한 줄이면 충분


    # 옵션/버튼 (session_state에 보존)
    # 옵션/버튼
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        threshold = st.number_input(
            "충족 임계(수량합)",
            min_value=1, max_value=5,
            value=st.session_state.get("threshold", 1),
            step=1,
            key="threshold"   # ✅ key만 주고, session_state에 대입하지 않음
        )
    with c2:
        export_flag = st.checkbox(
            "log.csv 저장",
            value=st.session_state.get("export_flag", True),
            key="export_flag" # ✅ 마찬가지로 대입 금지
        )
    with c3:
        analyze_clicked = st.button("분석하기", type="primary", key="analyze_btn")

    # c1, c2, c3 = st.columns([1, 1, 1])
    # with c1:
    #     st.session_state.threshold = st.number_input(
    #         "충족 임계(수량합)", min_value=1, max_value=5, value=st.session_state.get("threshold", 1), step=1, key="threshold"
    #     )
    # with c2:
    #     st.session_state.export_flag = st.checkbox(
    #         "log.csv 저장", value=st.session_state.get("export_flag", True), key="export_flag"
    #     )
    # with c3:
    #     analyze_clicked = st.button("분석하기", type="primary", key="analyze_btn")

    # ===== 분석 실행 =====
    if analyze_clicked:
        try:
            all_items_df_list = []
            total_counts = defaultdict(float)
            all_logs = []
            all_unmatched = []

            for slot in SLOTS:
                items_df, counts, log_df, unmatched = analyze_items_for_slot(
                    st.session_state.inputs.get(slot, ""), slot, df_food, nutrient_desc
                )
                if not items_df.empty:
                    items_df["날짜"] = d.isoformat()
                if not log_df.empty:
                    log_df["date"] = d.isoformat()
                all_items_df_list.append(items_df)
                for k, v in counts.items():
                    total_counts[k] += float(v or 0)
                all_logs.append(log_df)
                all_unmatched += unmatched

            items_df_all = (
                pd.concat(all_items_df_list, ignore_index=True)
                if all_items_df_list else
                pd.DataFrame(columns=["슬롯", "입력항목", "수량", "매칭식품", "등급", "태그", "날짜"])
            )
            logs_all = (
                pd.concat(all_logs, ignore_index=True)
                if all_logs else
                pd.DataFrame(columns=["timestamp", "date", "time", "slot", "입력항목", "수량", "매칭식품", "등급", "태그"])
            )

            # ✅ 결과를 session_state에 저장 (리런/새로고침에도 유지)
            st.session_state.last_items_df = items_df_all
            st.session_state.last_nutri_df = summarize_nutrients(
                dict(total_counts), df_food, nutrient_desc, threshold=int(st.session_state.threshold)
            )
            st.session_state.last_recs, st.session_state.last_combo = recommend_next_meal(
                dict(total_counts), df_food, nutrient_desc,
                # 필요 시 옵션 활성화:
                # tag_targets={'단백질': 2, '식이섬유': 2},
                # prefer_tags=['식이섬유','단백질'],
                # avoid_tags=['당','탄수화물'],
                # allowed_grades=('Safe','Caution'),
                # max_items=4
            )

            # ===== log.csv 저장 & 다운로드 =====
            if st.session_state.export_flag and not logs_all.empty:
                try:
                    # 기존 파일이 있으면 append, 없으면 생성
                    try:
                        prev = pd.read_csv(LOG_CSV)
                        merged = pd.concat([prev, logs_all], ignore_index=True)
                    except Exception:
                        merged = logs_all.copy()
                    merged.to_csv(LOG_CSV, index=False, encoding="utf-8-sig")
                    st.success(f"'{LOG_CSV}' 저장 완료")
                    with open(LOG_CSV, "rb") as f:
                        st.download_button("⬇️ log.csv 다운로드", data=f.read(), file_name="log.csv", mime="text/csv")
                except Exception as ex:
                    st.error(f"log.csv 저장/다운로드 실패: {ex}")

            # ===== food_db 업데이트 (미매칭 재료 추가) =====
            df_food_updated = append_unmatched_to_food_db(df_food, all_unmatched)
            try:
                # 태그리스트를 원래 CSV 포맷으로 되돌리기 (보기용 '태그(영양)'도 동기화)
                df_export = df_food_updated.copy()
                df_export["태그리스트"] = df_export["태그리스트"].apply(_parse_taglist_cell)
                df_export["태그(영양)"] = df_export["태그리스트"].apply(lambda lst: "/".join(lst))
                df_export.to_csv(FOOD_DB_UPDATED_CSV, index=False, encoding="utf-8-sig")
                st.success("미매칭 재료를 포함한 'food_db_updated.csv' 생성 완료")
                with open(FOOD_DB_UPDATED_CSV, "rb") as f:
                    st.download_button("⬇️ food_db_updated.csv 다운로드", data=f.read(), file_name="food_db_updated.csv", mime="text/csv")
            except Exception as ex:
                st.error(f"food_db 업데이트/다운로드 실패: {ex}")

        except Exception as e:
            st.error(f"분석 중 오류: {e}")

    # ===== 화면 표시: 분석 버튼을 안 눌러도 마지막 결과 유지해 보여주기 =====
    st.markdown("### 🍱 슬롯별 매칭 결과")
    if st.session_state.last_items_df is None or st.session_state.last_items_df.empty:
        st.info("매칭된 항목이 없습니다.")
    else:
        st.dataframe(
            st.session_state.last_items_df[["날짜", "슬롯", "입력항목", "수량", "매칭식품", "등급", "태그"]],
            use_container_width=True,
            height=min(420, 36 * (len(st.session_state.last_items_df) + 1))
        )

    st.markdown("### 🧭 영양 태그 요약 (충족/부족 + 한줄설명)")
    if st.session_state.last_nutri_df is None or st.session_state.last_nutri_df.empty:
        st.info("영양소 사전 또는 태그 정보가 비어 있습니다.")
    else:
        st.dataframe(
            st.session_state.last_nutri_df,
            use_container_width=True,
            height=min(420, 36 * (len(st.session_state.last_nutri_df) + 1))
        )

    st.markdown("### 🍽 다음 식사 제안 (부족 보완용)")
    if not st.session_state.last_recs:
        st.success("핵심 부족 영양소가 없습니다. 균형이 잘 맞았어요!")
    else:
        for r in st.session_state.last_recs:
            foods_text = ", ".join(r["추천식품"]) if r["추천식품"] else "(추천 식품 없음)"
            st.write(f"- **{r['부족영양소']}**: {r['설명']}")
            st.caption(f"  추천 식품: {foods_text}")
        if st.session_state.last_combo:
            st.info("간단 조합 제안: " + " / ".join(st.session_state.last_combo[:4]))





if __name__ == "__main__":
    if st is None:
        pass
    else:
        main()
