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
    # 보장 컬럼
    for c in ["식품", "등급", "태그(영양)"]:
        if c not in df.columns:
            df[c] = ""
    # 태그리스트 정규화
    if "태그리스트" not in df.columns:
        df["태그리스트"] = df["태그(영양)"].apply(_parse_tags_from_slash)
    else:
        df["태그리스트"] = [
            _ensure_taglist(row.get("태그리스트", None), row.get("태그(영양)", None))
            if isinstance(row, dict) else _ensure_taglist(df.loc[i, "태그리스트"], df.loc[i, "태그(영양)"])
            for i, row in enumerate([{}]*len(df))
        ]
    # 최소 컬럼 반환
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


def summarize_nutrients(nutrient_counts: Dict[str, float], df_food: pd.DataFrame, nutrient_desc: Dict[str, str], threshold: int = 1) -> pd.DataFrame:
    all_tags = sorted({t for tlist in df_food["태그리스트"].apply(_parse_taglist_cell) for t in tlist})
    rows = []
    for tag in all_tags:
        cnt = float(nutrient_counts.get(tag, 0))
        rows.append({
            "영양소": tag,
            "수량합": cnt,
            "상태": "충족" if cnt >= threshold else "부족",
            "한줄설명": nutrient_desc.get(tag, "")
        })
    return pd.DataFrame(rows).sort_values(["상태", "수량합", "영양소"], ascending=[True, False, True])


def recommend_next_meal(nutrient_counts: Dict[str, float], df_food: pd.DataFrame, nutrient_desc: Dict[str, str],
                        top_nutrients: int = 2, per_food: int = 4):
    """부족 영양소 중심 추천: Safe 식품 우선 + 간단 조합"""
    # 태그 우주 생성 시에도 안정적으로 리스트 파싱
    tag_universe = {tt for lst in df_food["태그리스트"].apply(_parse_taglist_cell) for tt in lst}
    tag_totals = {t: float(nutrient_counts.get(t, 0)) for t in tag_universe}
    lacking = [t for t, v in sorted(tag_totals.items(), key=lambda x: x[1]) if v < 1.0]
    lacking = lacking[:top_nutrients]

    suggestions = []
    for tag in lacking:
        pool = df_food[
            (df_food["등급"] == "Safe") &
            (df_food["태그리스트"].apply(lambda lst: tag in _parse_taglist_cell(lst)))
        ]
        foods = pool["식품"].dropna().astype(str).head(per_food).tolist()
        suggestions.append({
            "부족영양소": tag,
            "설명": nutrient_desc.get(tag, ""),
            "추천식품": foods
        })

    combo = []
    for s in suggestions:
        for f in s["추천식품"]:
            if f not in combo:
                combo.append(f)
            if len(combo) >= 4:
                break
        if len(combo) >= 4:
            break

    return suggestions, combo


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
    try:
        import streamlit as st
    except Exception as e:
        print("This script requires Streamlit to run the UI. Install with: pip install streamlit")
        sys.exit(1)

    st.set_page_config(page_title="슬롯별 식단 분석 · 다음 식사 제안", page_icon="🥗", layout="centered")
    st.title("🥗 슬롯별 식단 분석 · 다음 식사 제안")

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

    # 날짜 + 슬롯별 입력
    d = st.date_input("기록 날짜", value=date.today())
    inputs = {}
    cols = st.columns(1)
    with st.container():
        for slot in SLOTS:
            inputs[slot] = st.text_area(f"{slot}", height=70, placeholder=f"{slot}에 먹은 것 입력")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        threshold = st.number_input("충족 임계(수량합)", min_value=1, max_value=5, value=1, step=1)
    with c2:
        export_flag = st.checkbox("log.csv 저장", value=True)
    with c3:
        analyze_clicked = st.button("분석하기", type="primary")

    if analyze_clicked:
        try:
            all_items_df_list = []
            total_counts = defaultdict(float)
            all_logs = []
            all_unmatched = []

            for slot in SLOTS:
                items_df, counts, log_df, unmatched = analyze_items_for_slot(inputs.get(slot, ""), slot, df_food, nutrient_desc)
                if not items_df.empty:
                    items_df["날짜"] = d.isoformat()
                if not log_df.empty:
                    log_df["date"] = d.isoformat()
                all_items_df_list.append(items_df)
                for k, v in counts.items():
                    total_counts[k] += float(v or 0)
                all_logs.append(log_df)
                all_unmatched += unmatched

            items_df_all = pd.concat(all_items_df_list, ignore_index=True) if all_items_df_list else pd.DataFrame(columns=["슬롯","입력항목","수량","매칭식품","등급","태그","날짜"])
            logs_all = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame(columns=["timestamp","date","time","slot","입력항목","수량","매칭식품","등급","태그"])

            st.markdown("### 🍱 슬롯별 매칭 결과")
            if items_df_all.empty:
                st.info("매칭된 항목이 없습니다.")
            else:
                st.dataframe(items_df_all[["날짜","슬롯","입력항목","수량","매칭식품","등급","태그"]], use_container_width=True, height=min(420, 36 * (len(items_df_all) + 1)))

            st.markdown("### 🧭 영양 태그 요약 (충족/부족 + 한줄설명)")
            nutri_df = summarize_nutrients(dict(total_counts), df_food, nutrient_desc, threshold=int(threshold))
            if nutri_df.empty:
                st.info("영양소 사전 또는 태그 정보가 비어 있습니다.")
            else:
                st.dataframe(nutri_df, use_container_width=True, height=min(420, 36 * (len(nutri_df) + 1)))

            st.markdown("### 🍽 다음 식사 제안 (부족 보완용)")
            recs, combo = recommend_next_meal(dict(total_counts), df_food, nutrient_desc, top_nutrients=2, per_food=4)
            if not recs:
                st.success("핵심 부족 영양소가 없습니다. 균형이 잘 맞았어요!")
            else:
                for r in recs:
                    foods_text = ", ".join(r["추천식품"]) if r["추천식품"] else "(추천 식품 없음)"
                    st.write(f"- **{r['부족영양소']}**: {r['설명']}")
                    st.caption(f"  추천 식품: {foods_text}")
                if combo:
                    st.info("간단 조합 제안: " + " / ".join(combo[:4]))

            # ===== log.csv 저장 & 다운로드 =====
            if export_flag and not logs_all.empty:
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


if __name__ == "__main__":
    if st is None:
        pass
    else:
        main()
