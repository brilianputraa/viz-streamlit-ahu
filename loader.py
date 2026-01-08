import os, glob, io, re, json
import pandas as pd
import chardet
from datetime import timedelta, datetime
import numpy as np
import streamlit as st
import hashlib 

# ===== 경로 =====
# Environment variable overrides (Windows paths as default for compatibility)
HISTORY_DIR = os.getenv("AHU_HISTORY_DIR", r"C:\Users\User\Desktop\history")
RESULT_BASE = os.getenv("AHU_RESULT_BASE", r"C:\Users\User\Desktop\ahu_app_results")

RAW_DIR   = os.path.join(RESULT_BASE, "raw_results")
FINAL_DIR = os.path.join(RESULT_BASE, "final_results")
OA_DIR    = os.path.join(RESULT_BASE, "oa_results")

for d in [RESULT_BASE, RAW_DIR, FINAL_DIR, OA_DIR]:
    os.makedirs(d, exist_ok=True)

META_FILE = os.path.join(FINAL_DIR, "processed_files.json")  # 파일명+mtime 메타

# ===== 단가 (연도별) =====
단가_딕셔너리 = {
    2022: {"냉수단가": 295, "증기단가": 52300, "전기단가": 119},
    2023: {"냉수단가": 299, "증기단가": 57500, "전기단가": 154},
    2024: {"냉수단가": 304, "증기단가": 61600, "전기단가": 168},
    2025: {"냉수단가": 307, "증기단가": 65000, "전기단가": 182}
}
def get_단가(year):
    return 단가_딕셔너리.get(year, {"냉수단가": 300, "증기단가": 60000, "전기단가": 150})

from common import (
    get_단가, 항목_열량맵핑,
    서플라이팬용량, 프로세스팬용량, 배기팬용량,
    기어모터용량, 로터모터용량,
    CDU용량, HEATER용량,
    건식제습형_공조기
)

def _resolve_device_and_power(ahu_base: str, tag: str):
    """
    tag 예: SFST, SFST1, SFST2, RFST1, EFST2, PC_SFST, COMPSS1, CDUSS, EHSS2 ...
    리턴: (장치그룹, kW 용량)
    """
    t = str(tag).upper()

    # 장치 그룹
    if "SF" in t: group = "SF"
    elif "EF" in t: group = "EF"
    elif "RF" in t: group = "RF"
    elif "CDU" in t or "COMP" in t: group = "CDU/COMP"
    elif "EH" in t or "HT" in t: group = "EH"
    else: group = "기타"

    # 번호 포함된 키 추출 (SFST2, RFST1, COMPSS1, CDUSS, EHSS2 ...)
    m = re.search(r'(SFST\d*|SFSS\d*|RFST\d*|RFSS\d*|EFST\d*|EFSS\d*|COMPSS\d*|COMP\d*|CDUSS|CDU|EHSS\d*|EH|HTSS|HT)', t)
    device_key = m.group(1) if m else t

    # 번호 포함 키 우선 조회 → 없으면 기본키로 fallback
    if group == "SF":
        kw = 서플라이팬용량.get((ahu_base, device_key))
        if kw is None: kw = 서플라이팬용량.get(ahu_base)
    elif group == "RF":
        kw = 프로세스팬용량.get((ahu_base, device_key))
        if kw is None: kw = 프로세스팬용량.get(ahu_base)
    elif group == "EF":
        kw = 배기팬용량.get((ahu_base, device_key))
        if kw is None: kw = 배기팬용량.get(ahu_base)
    elif group == "CDU/COMP":
        kw = CDU용량.get((ahu_base, device_key))
        if kw is None: kw = CDU용량.get((ahu_base, "COMP"))
    elif group == "EH":
        kw = HEATER용량.get((ahu_base, device_key))
        if kw is None: kw = HEATER용량.get((ahu_base, "HTSS"))
    else:
        kw = 0

    return group, float(kw or 0.0)


def 보간_열량계산(df, 항목명, 최대열량, 이상값탐지=True, midnight_only=True):
    df = df.sort_values("datetime").reset_index(drop=True)

    # 1. 시간 간격 계산
    df["시간간격"] = df["datetime"].diff().dt.total_seconds() / 3600

    # 2. 00:00 누락 보간 (옵션)
    if midnight_only:
        new_rows = []
        for day in pd.date_range(df["datetime"].min().normalize(),
                                 df["datetime"].max().normalize()):
            midnight = day
            if midnight not in df["datetime"].values:
                before = df[df["datetime"] < midnight].tail(1)
                after = df[df["datetime"] > midnight].head(1)
                if not before.empty and not after.empty:
                    val = before["값"].iloc[0] + (after["값"].iloc[0] - before["값"].iloc[0]) * (
                        (midnight - before["datetime"].iloc[0]) /
                        (after["datetime"].iloc[0] - before["datetime"].iloc[0])
                    )
                    new_rows.append({"datetime": midnight, "공조기": df["공조기"].iloc[0],
                                     "항목명": 항목명, "값": val})
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True).sort_values("datetime")

    # 3. 유효한 시간 간격만 남기기
    df = df[(df["시간간격"] > 0) & (df["시간간격"] <= 12)].copy()

    # 4. 열량 계산 (trapezoid 방식)
    v1, v2 = df["값"].shift(1), df["값"]
    df["열량_kWh"] = ((v1 + v2) / 2) * 최대열량 * df["시간간격"] / 100 / 860

    # 5. 이상값 탐지
    if 이상값탐지:
        df.loc[df["열량_kWh"] > 300, "열량_kWh"] = np.nan

    # 6. 비용 계산 (단가는 연도별로 나중에 적용, 일단 placeholder)
    df["비용(원)"] = np.nan

    return df


# ===== 공조기 타입 =====
건식제습형_공조기 = {"AHU03", "AHU07", "AHU09", "AHU11", "AHU14", "AHU021", "AHU023", "AHU025", "AHU026"}

# ===== CSV 로더 =====
def _safe_read_bytes(path: str):
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

def _detect_encoding(b: bytes):
    det = chardet.detect(b)
    return det.get("encoding") or "utf-8"

def read_csv_fast(path: str) -> pd.DataFrame:
    raw = _safe_read_bytes(path)
    if not raw:
        return pd.DataFrame()
    enc = _detect_encoding(raw)
    try:
        iter_csv = pd.read_csv(
            io.BytesIO(raw),
            encoding=enc,
            chunksize=500_000,
            low_memory=False,
            dtype_backend="pyarrow",
        )
        return pd.concat(iter_csv, ignore_index=True)
    except Exception:
        return pd.DataFrame()

# ===== CSV 파서 =====
def parse_ahu_csv(path: str) -> pd.DataFrame:
    df = read_csv_fast(path)
    if df.empty:
        return df

    # --- 컬럼명 정규화 ---
    rename_map = {
        "Date": "datetime",
        "날짜": "datetime",
        "date": "datetime",
        "Value": "값",
        "value": "값",
        "POINT": "point",
        "Point": "point",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "datetime" not in df.columns:
        raise ValueError(f"{path} 파일에 datetime 컬럼이 없습니다.")

    # 🚨 상단 불필요 행 제거
    if "값" in df.columns:
        mask_valid = (
            df["datetime"].notna() & df["datetime"].astype(str).str.strip().ne("") &
            df["값"].notna() & df["값"].astype(str).str.strip().ne("")
        )
        df = df[mask_valid].copy()

    if df.empty:
        return df

    # datetime 문자열 처리
    df["datetime"] = df["datetime"].astype(str).str.strip()
    try:
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H%M", errors="coerce")
    except Exception:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df = df.dropna(subset=["datetime", "값"]).copy()

    # 값 숫자 변환
    df["값"] = pd.to_numeric(df["값"], errors="coerce")
    df = df.dropna(subset=["값"])

    # --- 공조기명 추출 ---
    fname = os.path.basename(path)
    fname_noext = os.path.splitext(fname)[0]
    ahu = re.sub(r"_\d+$", "", fname_noext.upper().split("-")[0])
    df["공조기"] = ahu

    # --- 항목명 추출 ---
    if "point" in df.columns:
        df["항목명"] = (
            df["point"].astype(str).str.upper()
            .str.replace(r"^AHU\d+_?", "", regex=True)
            .str.replace(r"\.PRESENTVALUE$", "", regex=True)
        )
    else:
        df["항목명"] = "UNKNOWN"

    # 🔧 여기부터 추가 (공백/변형 정리)
    df["항목명"] = df["항목명"].str.upper().str.strip()
    df["항목명"] = df["항목명"].str.replace(r"\s+(?=\d)", "", regex=True)   # 'SFST 1' -> 'SFST1'
    df["항목명"] = df["항목명"].str.replace(r"^AC_(CCV|HCV)$", r"\1", regex=True)  # 'AC_CCV'->'CCV'


    return df[["datetime", "공조기", "항목명", "값"]]


def parse_oa_csv(path: str) -> pd.DataFrame:
    df = read_csv_fast(path)
    if df.empty or df.shape[1] < 3:
        return pd.DataFrame()
    df = df.iloc[:, :3]
    df.columns = ["label", "datetime", "value"]
    df["datetime"] = pd.to_datetime(df["datetime"].astype(str), errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df_temp = df[df["label"].str.contains("OA_T", case=False, na=False)].rename(columns={"value": "외기온도"})
    df_humi = df[df["label"].str.contains("OA_H", case=False, na=False)].rename(columns={"value": "외기습도"})
    return pd.merge(
        df_temp[["datetime", "외기온도"]],
        df_humi[["datetime", "외기습도"]],
        on="datetime",
        how="outer",
    )

# ==== 장치 분류 세트 & 헬퍼 ====
# 코일 별칭
COIL_ALIASES = {
    "CCV": {"CCV", "AC_CCV", "PC_CCV"},
    "HCV": {"HCV", "AC_HCV", "DH_HCV"},   # DH_HCV는 스팀 제습 코일(별도 집계 유지)
}

# 전기부하만 여기 포함
MOTOR_SET = {
    "SFST","SFSS","EFST","EFSS","RFST",
    "AC_SFST","PC_SFST","OAU_SFST","AC_RFSS",
    "COMPST","CDU","CDUSST","COMP",
    "EHST","EHSS1","EHSS2","EHSS3",
    "DH_GMST",  # 제습 휠 기어모터
}

# 센서/표시치(비용·운전시간 생성 금지)
SENSOR_SET = {
    "RAT","RAH","AC_RAT","AC_RAH","DH_DEH","DH_TEMP",
    "AC_SAT","PC_SAT","AC_SAH","PC_HCV"
}

def _token_from_col(c: str):
    """wide 컬럼에서 장치 토큰만 뽑는 유틸"""
    if c.startswith("비용(원)_"):      return c.split("_", 1)[1]
    if c.endswith("_비용(원)"):        return c.rsplit("_", 1)[0]
    if c.startswith(("kWh_","kwh_")):  return c.split("_", 1)[1]
    if c.lower().endswith("_kwh"):     return c.rsplit("_", 1)[0]
    if c.startswith("운전시간(h)_"):    return c.split("_", 1)[1]
    return None


# ===== 에너지/비용 계산 =====
@st.cache_data
def calculate_final_from_raw(raw_df: pd.DataFrame, min_dt=None, max_dt=None) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    # ✅ 신규 구간만 잘라내기
    if min_dt is not None and max_dt is not None:
        raw_df = raw_df[(raw_df["datetime"] >= min_dt) & (raw_df["datetime"] <= max_dt)].copy()
    if raw_df.empty:
        return pd.DataFrame()

    raw_df = raw_df.copy()
    raw_df["연도"] = raw_df["datetime"].dt.year
    raw_df["시간"] = raw_df["datetime"].dt.floor("H")

    # 🔧 추가: 항목명 정규화(2차 방어)
    raw_df["항목명"] = raw_df["항목명"].astype(str).str.upper().str.strip()
    raw_df["항목명"] = raw_df["항목명"].str.replace(r"\s+(?=\d)", "", regex=True)
    raw_df["항목명"] = raw_df["항목명"].str.replace(r"^AC_(CCV|HCV)$", r"\1", regex=True)


    results = []

    # ===== 세부항목별 계산 =====
    for (ahu, year, 항목), grp in raw_df.groupby(["공조기", "연도", "항목명"]):
        grp = grp.sort_values("datetime")
        grp["dt_h"] = grp["datetime"].diff().dt.total_seconds() / 3600
        grp = grp[(grp["dt_h"] > 0) & (grp["dt_h"] <= 5)].copy()
        if grp.empty:
            continue

        hourly = grp.groupby("시간", as_index=False).agg({"dt_h": "sum", "값": "mean"})
        hourly["연도"] = hourly["시간"].dt.year
        단가 = get_단가(hourly["연도"].iloc[0])
        ahu_base = ahu.split("_")[0]

        item = 항목
        # === kWh / 비용 계산 ===
        if item in ("CCV", "PC_CCV"):
            최대열량 = 항목_열량맵핑[항목].get(ahu_base, 0)
            hourly["kWh"] = hourly["값"] * 최대열량 * 0.01 * hourly["dt_h"] / 860
            hourly["비용(원)"] = hourly["kWh"] * 단가["냉수단가"] * 860 / (2.3 * 4.187 * 1000)

            hourly["공조기"] = ahu_base
            hourly["항목명"] = 항목
            results.append(hourly[["공조기","시간","연도","항목명","kWh","비용(원)"]])

        elif item == "DH_HCV" or item.startswith("DH_HCV"):
            최대열량 = 항목_열량맵핑[항목].get(ahu_base, 0)
            hourly["kWh"] = hourly["값"] * 최대열량 * 0.01 * hourly["dt_h"] / 860
            hourly["비용(원)"] = hourly["kWh"] * 단가["증기단가"] * 860 / (495 * 0.4 * 1000)

            hourly["공조기"] = ahu_base
            hourly["항목명"] = 항목
            results.append(hourly[["공조기","시간","연도","항목명","kWh","비용(원)"]])

        elif item in ("HCV", "AC_HCV"):
            최대열량 = 항목_열량맵핑[항목].get(ahu_base, 0)
            hourly["kWh"] = hourly["값"] * 최대열량 * 0.01 * hourly["dt_h"] / 860
            hourly["비용(원)"] = hourly["kWh"] * 단가["증기단가"] * 860 / (540 * 0.4 * 1000)

            hourly["공조기"] = ahu_base
            hourly["항목명"] = 항목
            results.append(hourly[["공조기","시간","연도","항목명","kWh","비용(원)"]])

        elif any(key in item for key in ["SFST","PC_SFST","RFST","EFST","COMP","CDU","EH","HT","DH_EFST","DH_GMST"]):
            # ⬇️ 여길 전부 아래 코드로 교체
            ahu_base = ahu.split("_")[0]

            # 시계열 적분: state(0/1) × kW × dt_h  (사다리꼴)
            grp = grp.sort_values("datetime").copy()
            grp["dt_h"] = grp["datetime"].diff().dt.total_seconds() / 3600
            grp = grp[(grp["dt_h"] > 0) & (grp["dt_h"] <= 12)]

            # 태그별(kW) 해석
            _, kw_cap = _resolve_device_and_power(ahu_base, 항목)

            v1 = grp["값"].shift(1).fillna(0)
            v2 = grp["값"].fillna(0)
            state_avg = ((v1 + v2) / 2.0).fillna(0)

            grp["kWh_seg"] = state_avg * kw_cap * grp["dt_h"]
            grp["run_seg"] = state_avg * grp["dt_h"]


            hourly = grp.groupby("시간", as_index=False).agg(
                kWh=("kWh_seg","sum"),
                운전시간_h=("run_seg","sum")  # ← dt_h 말고 run_seg 합산
            )
            hourly.rename(columns={"운전시간_h": "운전시간(h)"}, inplace=True)

            hourly["연도"] = hourly["시간"].dt.year
            hourly["비용(원)"] = hourly["kWh"] * 단가["전기단가"]
            hourly["공조기"] = ahu_base
            hourly["항목명"] = 항목
            results.append(hourly)
        

    if not results:
        return pd.DataFrame()

    detail_df = pd.concat(results, ignore_index=True)

    # ===== 2. 큰 항목별 (냉수/스팀/전력) 합산 =====
    group_map = {
    "냉수": ["CCV", "PC_CCV", "AC_CCV"],
    "스팀": ["HCV", "DH_HCV", "AC_HCV"],
    "전력": [
        "SFST", "SFST1", "SFST2", 
        "SFSS", "SFSS1", "SFSS2", # <- 이 항목들이 누락되었을 수 있습니다.
        "RFST", "RFST1", "RFST2", "RFSS", "RFSS1", "RFSS2",
        "EFST", "EFST1", "EFST2", "EFSS", "EFSS1", "EFSS2",
        "PC_SFST", "AC_SFST", "AC_RFSS", "OAU_SFST",
        "CDU", "CDUSST", "COMP", "COMPST", "COMPSS1", "COMPSS2",
        "EHST", "EHSS1", "EHSS2", "EHSS3",
        "DH_EFST", "DH_GMST"
    ]
}

    big_rows = []
    for (공조기, 시간), grp in detail_df.groupby(["공조기", "시간"]):
        연도 = grp["연도"].iloc[0]
        for big, 세부 in group_map.items():
            sub = grp[grp["항목명"].isin(세부)]
            if sub.empty:
                continue
            kWh = sub["kWh"].sum()
            비용 = sub["비용(원)"].sum()
            big_rows.append(
                {
                    "공조기": 공조기,
                    "시간": 시간,
                    "연도": 연도,
                    "항목명": big,
                    "kWh": kWh,
                    "비용(원)": 비용,
                }
            )

    big_df = pd.DataFrame(big_rows)

    # ===== 3. 총합 (냉수+스팀+전력) =====
    total_rows = []
    for (공조기, 시간), grp in big_df.groupby(["공조기", "시간"]):
        연도 = grp["연도"].iloc[0]
        kWh = grp["kWh"].sum()
        비용 = grp["비용(원)"].sum()
        total_rows.append(
            {
                "공조기": 공조기,
                "시간": 시간,
                "연도": 연도,
                "총합_kWh": kWh,
                "총합_비용": 비용,
            }
        )
    total_df = pd.DataFrame(total_rows)

    # ===== 4. pivot으로 열 단위 정리 =====
    values_cols = [c for c in ["kWh", "비용(원)", "평균 개도율(%)", "운전시간(h)"] if c in detail_df.columns]

    pivot_detail = detail_df.pivot_table(
        index=["공조기", "시간", "연도"],
        columns="항목명",
        values=values_cols,
        aggfunc="sum",
    )
    pivot_detail.columns = [f"{c1}_{c2}" for c1, c2 in pivot_detail.columns]
    pivot_detail = pivot_detail.reset_index()


    pivot_big = big_df.pivot_table(
    index=["공조기", "시간", "연도"],
    columns="항목명",
    values=["kWh", "비용(원)"],
    aggfunc="sum",
    )
    pivot_big.columns = [f"{c1}_{c2}" for c1, c2 in pivot_big.columns]
    pivot_big = pivot_big.reset_index()


    final_df = pivot_detail.merge(pivot_big, on=["공조기", "시간", "연도"], how="outer")
    if not total_df.empty:
        final_df = final_df.merge(total_df, on=["공조기", "시간", "연도"], how="outer")

    final_df["datetime"] = final_df["시간"]
    final_df = final_df.drop(columns=[c for c in final_df.columns if re.match(r'^(CCV|HCV).*운전시간', c)], errors='ignore')
    #  A) 센서 유래의 비용/운전시간 컬럼 제거 (혹시 생겼다면 방지차 한 번 더)
    bad_cols = [
        c for c in final_df.columns
        if (c.startswith("비용(원)_") or c.endswith("_비용(원)") or c.startswith("운전시간(h)_"))
        and (_token_from_col(c) in SENSOR_SET)
    ]
    final_df.drop(columns=bad_cols, inplace=True, errors="ignore")

    #  B) 전력/코일 비용 표준 컬럼 만들기 (보기용 레이블 통일)
    # 전력 합산 컬럼 표준화
    if "전력_비용(원)" in final_df.columns:
        pass  # 이미 있으면 그대로 사용
    elif "비용(원)_전력" in final_df.columns:
        final_df["전력_비용(원)"] = final_df["비용(원)_전력"]

    if "비용(원)_냉수" in final_df.columns:
        final_df["냉수_비용(원)"] = final_df["비용(원)_냉수"]
    if "비용(원)_스팀" in final_df.columns:
        final_df["스팀_비용(원)"] = final_df["비용(원)_스팀"]
    if "총합_비용" in final_df.columns and "총합_비용(원)" not in final_df.columns:
        final_df["총합_비용(원)"] = final_df["총합_비용"]

    #  C) 전력_비용(원) 없거나 전부 NaN이면 → 모터 비용 합/보정
    need_power_fill = ("전력_비용(원)" not in final_df.columns) or final_df["전력_비용(원)"].isna().all()

    if need_power_fill:
        # 모터 비용 컬럼 합산 시도
        motor_cost_cols = [
            c for c in final_df.columns
            if c.startswith("비용(원)_") and (_token_from_col(c) in MOTOR_SET)
        ]
        if motor_cost_cols:
            final_df[motor_cost_cols] = final_df[motor_cost_cols].apply(pd.to_numeric, errors="coerce")
            final_df["전력_비용(원)"] = final_df[motor_cost_cols].sum(axis=1, min_count=1)

    # 비용 합산도 없으면 kWh × 단가로 보정
    if ("전력_비용(원)" not in final_df.columns) or final_df["전력_비용(원)"].isna().all():
        motor_kwh_cols = [
            c for c in final_df.columns
            if (_token_from_col(c) in MOTOR_SET) and (c.startswith("kWh_") or c.endswith("_kWh"))
        ]
        if motor_kwh_cols:
            final_df[motor_kwh_cols] = final_df[motor_kwh_cols].apply(pd.to_numeric, errors="coerce")
            kwh_sum = final_df[motor_kwh_cols].sum(axis=1, min_count=1)
            # 전기단가는 연도별 적용
            price = final_df["연도"].map(lambda y: get_단가(int(y))["전기단가"])
            final_df["전력_비용(원)"] = kwh_sum * price

    #  D) 혹시 모호한 컬럼 중복이 생겼다면 정리(선택)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    # === (=여기까지 추가)= ==========================================
    # 불필요한 계산 컬럼 제거
    drop_cols = ["DH_DEH_kWh", "DH_HCV_운전시간(h)", "PC_CCV_운전시간(h)"]
    final_df.drop(columns=[c for c in drop_cols if c in final_df.columns], inplace=True, errors="ignore")

    return final_df

# ===== 파일 시그니처 (파일명 + mtime) =====
def _get_files_signature(folder: str) -> dict:
    sig = {}
    for path in glob.glob(os.path.join(folder, "*.csv")):
        try:
            fname = os.path.basename(path)
            sig[fname] = os.path.getmtime(path)  # float (mtime)
        except FileNotFoundError:
            continue
    return sig

def load_processed_files() -> dict:
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: float(v) for k, v in data.items()}
        except Exception:
            return {}
    return {}

@st.cache_data
def load_oa_daily():
    """
    OA 일평균 parquet(oa_daily_*.parquet) 로드
    반환: datetime(자정), 외기온도, 외기습도
    """
    files = glob.glob(os.path.join(OA_DIR, "oa_daily_*.parquet"))
    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in files], ignore_index=True)

    # 기존 daily 파일은 'date' 컬럼 기준이므로 자정 datetime으로 변환
    # (standardize_oa가 datetime을 기대하므로 여기서 맞춰줌)
    if "date" in df.columns and "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        df = df.drop(columns=["date"])

    # 숫자화 방어
    for c in ["외기온도", "외기습도"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 정렬/중복 정리
    df = df.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates(subset=["datetime"])
    return df[["datetime", "외기온도", "외기습도"]]


def save_processed_files(signatures: dict):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(signatures, f, ensure_ascii=False, indent=2)


# ===== 메인 (증분 스캔 + 저장) =====
def scan_and_update(progress_callback=None):
    # 현재 파일 시그니처 (파일명 + mtime)
    current_sig = _get_files_signature(HISTORY_DIR)

    # 기존 시그니처 로드
    processed_sig = load_processed_files()

    # 변경된 파일만 추리기
    files_to_process = [
        os.path.join(HISTORY_DIR, fname)
        for fname, mtime in current_sig.items()
        if processed_sig.get(fname) != mtime
    ]

    if not files_to_process:
        print("⚠️ 새로운/변경된 파일 없음 → parquet만 로드", flush=True)
        # 변경된 파일이 없을 때
        return load_final_results(), load_oa_daily(), load_oa_results()


    raw_results, oa_results = [], []
    for i, path in enumerate(files_to_process):
        if progress_callback:
            progress_callback(i + 1, len(files_to_process), os.path.basename(path))
        base = os.path.basename(path).upper()

        if base.startswith("AHU"):
            df = parse_ahu_csv(path)
            if not df.empty:
                processed_rows = []
                for 항목명, grp in df.groupby("항목명"):
                    ahu_base = grp["공조기"].iloc[0]
                    최대열량 = 항목_열량맵핑.get(항목명, {}).get(ahu_base, 0)

                    # RAW에서는 보간만 (열량/비용 X)
                    grp = 보간_열량계산(
                        grp, 항목명, 최대열량,
                        이상값탐지=False,
                        midnight_only=True
                    )
                    grp = grp[["datetime", "공조기", "항목명", "값"]]
                    processed_rows.append(grp)

                df = pd.concat(processed_rows, ignore_index=True)
                raw_results.append(df)

        elif base.startswith("OA"):
            df = parse_oa_csv(path)
            if not df.empty:
                oa_results.append(df)

    # ===== AHU parquet 저장 (증분) =====
    if raw_results:
        ahu_raw_df = pd.concat(raw_results, ignore_index=True)

        for ahu, grp in ahu_raw_df.groupby("공조기"):
            out_path_raw   = os.path.join(RAW_DIR,   f"analysis_results_{ahu}.parquet")
            out_path_final = os.path.join(FINAL_DIR, f"final_analysis_{ahu}.parquet")

            # ✅ 이번 배치(새로 들어온 데이터)의 시간 범위만 기록
            new_min = grp["datetime"].min()
            new_max = grp["datetime"].max()

            # ---------- RAW 병합 ----------
            if os.path.exists(out_path_raw):
                old_lt = pd.read_parquet(out_path_raw, engine="pyarrow",
                                        filters=[("datetime", "<", new_min)])
                old_gt = pd.read_parquet(out_path_raw, engine="pyarrow",
                                        filters=[("datetime", ">", new_max)])
                combined = pd.concat([old_lt, grp, old_gt], ignore_index=True)
            else:
                combined = grp

            combined = (combined
                        .drop_duplicates(subset=["datetime","공조기","항목명","값"], keep="last")
                        .sort_values("datetime"))
            combined.to_parquet(out_path_raw, index=False, engine="pyarrow")

            # ---------- FINAL 계산 ----------
            # ⬅️ 여기! 전체가 아니라 '새 구간'만 계산
            final_delta = calculate_final_from_raw(combined, min_dt=new_min, max_dt=new_max)

            if not final_delta.empty:
                if os.path.exists(out_path_final):
                    old_lt = pd.read_parquet(out_path_final, engine="pyarrow",
                                            filters=[("datetime", "<", new_min)])
                    old_gt = pd.read_parquet(out_path_final, engine="pyarrow",
                                            filters=[("datetime", ">", new_max)])
                    old_final = pd.concat([old_lt, old_gt], ignore_index=True)
                    final_df = pd.concat([old_final, final_delta], ignore_index=True)
                    final_df = (final_df
                                .drop_duplicates(subset=["시간","공조기"], keep="last")
                                .sort_values("datetime"))
                else:
                    final_df = final_delta

                final_df = final_df.replace({pd.NA: np.nan})
                final_df.to_parquet(out_path_final, index=False, engine="pyarrow")


    else:
        print("⚠️ RAW parquet 저장 안됨 (raw_results 비어있음)", flush=True)

    # ===== OA parquet 저장 =====
    if oa_results:
        oa_raw_df = pd.concat(oa_results, ignore_index=True).drop_duplicates(subset=["datetime"])

        for year, grp in oa_raw_df.groupby(oa_raw_df["datetime"].dt.year):
            out_path_raw = os.path.join(OA_DIR, f"oa_results_{year}.parquet")
            if os.path.exists(out_path_raw):
                old_outside = pd.read_parquet(
                    out_path_raw, engine="pyarrow",
                    filters=[("datetime", "<", grp["datetime"].min())]
                )
                old_outside2 = pd.read_parquet(
                    out_path_raw, engine="pyarrow",
                    filters=[("datetime", ">", grp["datetime"].max())]
                )
                old = pd.concat([old_outside, old_outside2], ignore_index=True)
                grp = pd.concat([old, grp], ignore_index=True).drop_duplicates(subset=["datetime"])

            grp.to_parquet(out_path_raw, index=False, engine="pyarrow")

            # DAILY 저장
            daily = grp.copy()
            daily["date"] = daily["datetime"].dt.date
            daily_avg = daily.groupby("date", as_index=False)[["외기온도", "외기습도"]].mean(numeric_only=True)
            for c in ["외기온도", "외기습도"]:
                daily_avg[c] = pd.to_numeric(daily_avg[c], errors="coerce").round().astype("Int64")

            out_path_daily = os.path.join(OA_DIR, f"oa_daily_{year}.parquet")
            if os.path.exists(out_path_daily):
                old_outside = pd.read_parquet(
                    out_path_daily, engine="pyarrow",
                    filters=[("date", "<", daily_avg["date"].min())]
                )
                old_outside2 = pd.read_parquet(
                    out_path_daily, engine="pyarrow",
                    filters=[("date", ">", daily_avg["date"].max())]
                )
                old_daily = pd.concat([old_outside, old_outside2], ignore_index=True)
                daily_avg = pd.concat([old_daily, daily_avg], ignore_index=True).drop_duplicates(subset=["date"], keep="last")

            daily_avg.to_parquet(out_path_daily, index=False, engine="pyarrow")

    else:
        print("⚠️ OA parquet 저장 안됨 (oa_results 비어있음)", flush=True)

    save_processed_files(current_sig)
    print("DEBUG: processed_files.json 갱신 완료", flush=True)

    final_files = glob.glob(os.path.join(FINAL_DIR, "final_analysis_AHU*.parquet"))
    final_df = pd.DataFrame()
    if final_files:
        final_df = pd.concat([pd.read_parquet(f) for f in final_files])
        final_df["datetime"] = pd.to_datetime(final_df["datetime"])

    oa_results_df = load_oa_results()   # 고해상도
    oa_daily_df   = load_oa_daily()     # 일평균

    # ⬇️ 두 번째 리턴값을 이제 daily로!
    return final_df, oa_daily_df, oa_results_df


# ===== detail 로더 =====
def load_ahu_detail(ahu_name: str) -> pd.DataFrame:
    """
    특정 공조기의 detail parquet 읽기
    """
    exact = os.path.join(RAW_DIR, f"analysis_results_{ahu_name}.parquet")
    if not os.path.exists(exact):
        return pd.DataFrame()
    return pd.read_parquet(exact, engine="pyarrow")

# ===== 로더 =====
@st.cache_data
def load_final_results():
    """
    최종 집계(FINAL) parquet을 모두 읽어와 하나의 DataFrame으로 합침
    (전체 데이터 로드가 필요한 경우 사용)
    """
    files = glob.glob(os.path.join(FINAL_DIR, "final_analysis_AHU*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat(
        [pd.read_parquet(f, engine="pyarrow") for f in files],
        ignore_index=True
    )

@st.cache_data
def load_final_by_ahu(ahu_name: str) -> pd.DataFrame:
    """
    특정 공조기(AHU)의 최종 집계(FINAL) parquet만 읽어옴
    """
    path = os.path.join(FINAL_DIR, f"final_analysis_{ahu_name}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")

def load_detail_results(force_recalc=False):
    """
    RAW detail parquet 읽기 → 기본적으로는 사용 안 하고
    필요할 때만 호출
    """
    files = glob.glob(os.path.join(RAW_DIR, "analysis_results_AHU*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat(
        [pd.read_parquet(f, engine="pyarrow") for f in files],
        ignore_index=True
    )

@st.cache_data
def load_oa_results():
    """
    OA parquet → 항상 불러오기
    """
    files = glob.glob(os.path.join(OA_DIR, "oa_results_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat(
        [pd.read_parquet(f, engine="pyarrow") for f in files],
        ignore_index=True
    )

# ===== 유틸 함수 =====
def get_items_from_final(final_df: pd.DataFrame):
    """
    FINAL parquet에서 항목명 목록 추출
    (예: kWh_CCV → CCV, 비용(원)_HCV → HCV)
    """
    if final_df.empty:
        return []

    items = []
    for col in final_df.columns:
        if "_" in col and not col.startswith("총합"):
            items.append(col.split("_")[-1])
    return sorted(set(items))


def update_history_results(progress_callback=None):
    # 호환용 → 내부적으로 scan_and_update 호출
    return scan_and_update(progress_callback)

# ===== 호환용 함수 (app2.py와 연결) =====
def load_or_calculate_results():
    return load_final_results()