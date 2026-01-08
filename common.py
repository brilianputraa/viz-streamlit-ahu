import re, os
import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import chardet
from datetime import timedelta

# ✅ 최대 열량 하드코딩 (전체 AHU 값 반영)
냉수최대열량 = {
    "AHU01": 599000, "AHU02": 197000, "AHU03": 200000, "AHU04": 127000, "AHU05": 213000,
    "AHU06": 39000, "AHU07": 83000, "AHU08": 120000, "AHU09": 86100, "AHU10": 74000,
    "AHU11": 156000, "AHU12": 67500, "AHU13": 43000, "AHU14": 193000, "AHU15": 116000,
    "AHU16": 91000, "AHU17": 162000, "AHU18": 15000, "AHU19": 230000, "AHU20": 84000,
    "AHU21": 89000, "AHU22": 182000, "AHU23": 107800, "AHU24": 55400, "AHU25": 66000,
    "AHU26": 193000, "AHU32": 143000, "AHU33": 60000, "AHU36": 12814, "AHU37": 144000,
    "AHU40": 7393, "AHU41": 3845, "AHU42": 11363,
    "AHU43": 6135, "AHU44": 144000
}
증기최대열량 = {
    "AHU01": 393000, "AHU02": 85000, "AHU03": 157000, "AHU04": 89000, "AHU05": 157000,
    "AHU06": 15000, "AHU07": 23500, "AHU08": 126000, "AHU09": 64900, "AHU10": 34000,
    "AHU11": 113800, "AHU12": 47000, "AHU13": 28000, "AHU14": 190000, "AHU15": 114000,
    "AHU16": 87000, "AHU17": 126300, "AHU18": 11000, "AHU19": 139200, "AHU20": 93000,
    "AHU21": 73000, "AHU22": 102000, "AHU23": 105800, "AHU24": 29000, "AHU25": 60800,
    "AHU26": 190000, "AHU32": 90000, "AHU33": 20000, "AHU36": 59945, 
    "AHU40": 19555, "AHU41": 12915, "AHU42": 28493, "AHU43": 18460, "AHU44": 60000
}

PC_CCV_열량 = {
    "AHU03": 164000, "AHU07": 70000, "AHU09": 67000, "AHU11": 160000, "AHU14": 196000,"AHU21": 92000, "AHU23": 108600, "AHU25": 57600, "AHU26": 160000
}
DH_HCV_열량 = {
    "AHU03": 120000, "AHU07": 80000, "AHU09": 75800, "AHU11": 129000, "AHU14": 124000,"AHU21": 56800, "AHU23": 65400, "AHU25": 32800, "AHU26": 129000
}


항목_열량맵핑 = {
    "CCV": 냉수최대열량,
    "HCV": 증기최대열량,
    "PC_CCV": PC_CCV_열량,
    "DH_HCV": DH_HCV_열량,
    "전력": {}
}


서플라이팬용량 = {
    "AHU01": 44, ("AHU02","SFST1"): 15, ("AHU02","SFST2"): 15, "AHU03": 22, "AHU04": 30, ("AHU05","SFST1"): 37, ("AHU05","SFST2"): 37, "AHU06": 11, "AHU07": 18.5, "AHU08": 22, 
    "AHU09": 11, "AHU10": 18.5, "AHU11": 18.5, "AHU12": 7.5, "AHU13": 5.5, ("AHU14","SFST1"): 11, ("AHU14","SFST2"): 11, "AHU15": 15, "AHU16": 15, 
    "AHU19": 22, "AHU20": 18.5, "AHU21": 11, "AHU22": 22, "AHU23": 15, "AHU24": 11, "AHU25": 11,
    ("AHU26","SFST1"): 11, ("AHU26","SFST2"): 11, "AHU32": 7.5, "AHU33": 11, "AHU36": 11, "AHU37": 15, "AHU40": 0, "AHU41": 0, 
    "AHU42": 11, "AHU43": 7.5, "AHU44": 7.5
}

프로세스팬용량 = {
    "AHU03": 5.5, ("AHU06", "OAU_SFST"): 11, "AHU07": 2.2, "AHU09": 5.5, ("AHU10", "OAU_SFST"): 11, "AHU11": 7.5, ("AHU13", "OAU_SFST"): 11, 
    "AHU14": 7.5, ("AHU19", "RFST"): 18.5, 
    "AHU21": 3.7, "AHU23": 3.7, "AHU25": 2.2, "AHU26": 5.5, "AHU36": 3.2, "AHU37": 11, "AHU42": 3.7, "AHU43": 2.2
}   

배기팬용량 = {
    "AHU03": 2.2, "AHU07": 0.75, "AHU09": 2.2, "AHU11": 3.7, "AHU14": 3.7, "AHU21": 2.2, "AHU23": 2.2, "AHU25": 1.5, "AHU26": 3.7
}

기어모터용량 = {
    "AHU03": 1, "AHU07": 1, "AHU09": 1, "AHU11": 1, "AHU14": 1, "AHU21": 2.2, "AHU23": 2.2, "AHU25": 1.5, "AHU26": 3.7
}

로터모터용량 = {
    "AHU03": 1, "AHU07": 1, "AHU09": 1, "AHU11": 1, "AHU14": 1, "AHU21": 2.2, "AHU23": 2.2, "AHU25": 1.5, "AHU26": 3.7
}

CDU용량 = {
    ("AHU06", "OAU_CDUSS"): 15.0, ("AHU10", "OAU_CDUSS"): 15.0, ("AHU13", "OAU_CDUSS"): 15.0, ("AHU19","COMPSS1"): 10.0, ("AHU19","COMPSS2"): 10.0
}
HEATER용량 = {
    ("AHU06", "OAU_HTSS"): 2.0, ("AHU10", "OAU_HTSS"): 2.0, ("AHU13", "OAU_HTSS"): 2.0, ("AHU19", "EHSS1"): 1.0, ("AHU19", "EHSS2"): 1.0, ("AHU19", "EHSS3"): 1.0
}


건식제습형_공조기 = {"AHU03", "AHU07", "AHU09", "AHU11", "AHU14", "AHU21", "AHU23", "AHU25", "AHU26"}
냉각제습형_공조기 = {"AHU06", "AHU10", "AHU13", "AHU19"}

단가_딕셔너리 = {
    2022: {"냉수단가": 295, "증기단가": 52300, "전기단가": 119},
    2023: {"냉수단가": 299, "증기단가": 57500, "전기단가": 154},
    2024: {"냉수단가": 304, "증기단가": 61600, "전기단가": 168},
    2025: {"냉수단가": 307, "증기단가": 65000, "전기단가": 182}
}

def normalize_ahu_name(ahu: str) -> str:
    if not isinstance(ahu, str):
        return ahu
    ahu = ahu.upper().strip()
    m = re.match(r"AHU0*(\d+)", ahu)
    if not m:
        return ahu
    num = int(m.group(1))
    return f"AHU{num:02d}" if num < 10 else f"AHU{num}"


def _on_time_hours(series_datetime, series_state):
    """상태(1/0) × 구간시간(시간) 적분"""
    s = pd.DataFrame({"dt": series_datetime, "state": series_state}).sort_values("dt")
    s["dt_next"] = s["dt"].shift(-1)
    s["구간시"] = (s["dt_next"] - s["dt"]).dt.total_seconds() / 3600.0
    # 마지막 샘플의 dt_next는 NaT → 0 처리
    s["구간시"] = s["구간시"].fillna(0).clip(lower=0)
    return float((s["state"].fillna(0) * s["구간시"]).sum())

def get_motor_device_kwh(ahu_df: pd.DataFrame, ahu: str):
    """
    공조기 모터별 kWh 계산 (RAW/Parquet 모두 지원)
    - RAW(df에 '값' 존재): 장치 전력 시계열을 사다리꼴 적분
    - PARQUET(df에 'kWh'만 존재): 장치 전력 kWh를 합산
    """
    if ahu_df is None or ahu_df.empty:
        return 0.0, {}, {}

    # 어떤 컬럼에 장치명이 들어있는지
    col = "항목" if "항목" in ahu_df.columns else "항목명"

    # 장치 태그 패턴 (전력이라는 단어가 없어도 잡히도록)
    # SF/EF/RF (ST/SS/숫자) + OAU_/AC_/PC_ 접두, CDU/CDUSS, COMP/COMPSS, EH/EHSS/HT 계열까지 포함
    dev_regex = (
        r"(?:^|_)(?:OAU_|AC_|PC_)?(?:SF|EF|RF)(?:ST|SS)?\d*(?:$|_)"
        r"|(?:^|_)(?:CDU|CDUSS)(?:$|_)"
        r"|(?:^|_)(?:COMP|COMPSS\d*)(?:$|_)"
        r"|(?:^|_)(?:EH|EHSS\d*|HT|OAU_HTSS)(?:$|_)"
    )

    df = ahu_df.copy()
    df = df[df["공조기"].astype(str) == str(ahu)]
    if df.empty:
        return 0.0, {}, {}

    cand = df[df[col].astype(str).str.contains(dev_regex, regex=True, na=False)].copy()
    if cand.empty:
        return 0.0, {}, {}

    # 장치 그룹명을 6종으로 표준화
    def _device_group(s: str) -> str:
        s = str(s).upper()
        if "SF" in s:   return "SF"
        if "EF" in s:   return "EF"
        if "RF" in s:   return "RF"
        if "CDU" in s or "CDUSS" in s or "COMP" in s: return "CDU/COMP"
        if "EH" in s or "HT" in s: return "EH"
        return "기타"

    cand["장치"] = cand[col].map(_device_group)

    total_kwh = 0.0
    detail_kwh = {}
    detail_hours = {}

    if "값" in cand.columns:
        # RAW: kW × h 적분
        cand = cand.sort_values("datetime")
        cand["dt_h"] = cand.groupby("장치")["datetime"].diff().dt.total_seconds().div(3600)
        cand = cand[cand["dt_h"] > 0].copy()
        cand["값"] = pd.to_numeric(cand["값"], errors="coerce")

        # trapezoid
        v1 = cand.groupby("장치")["값"].shift(1)
        cand["kWh"] = ((v1 + cand["값"]) / 2.0) * cand["dt_h"]

        by_dev = cand.groupby("장치").agg(kwh=("kWh","sum"), hours=("dt_h","sum"))
        for dev, row in by_dev.iterrows():
            if row["kwh"] > 0:
                detail_kwh[dev] = float(row["kwh"])
                detail_hours[dev] = float(row["hours"])
                total_kwh += float(row["kwh"])

    elif "kWh" in cand.columns:
        # PARQUET: 이미 kWh가 있음 → 합산
        by_dev = cand.groupby("장치")["kWh"].sum()
        for dev, kwh in by_dev.items():
            if kwh > 0:
                detail_kwh[dev] = float(kwh)
                total_kwh += float(kwh)
        # 시간 정보가 없으니 hours는 생략/빈 dict 유지

    return total_kwh, detail_kwh, detail_hours


# common.py 상단에 필요하면 추가
# import streamlit as st

# common.py 상단 쪽

# common.py 상단 import 아래에 추가

# ── TAG 표준화 맵 (엑셀/CSV 실제 태그 → 내부 공통 이름) ──
TAG_MAP = {
    # 코일 밸브
    "CCV": "CCV",
    "AC_CCV": "CCV",
    "PC_CCV": "PC_CCV",
    "HCV": "HCV",
    "AC_HCV": "HCV",
    "DH_HCV": "DH_HCV",

    # 환기/급기 온습도
    "RAT": "RAT", "AC_RAT": "RAT",
    "RAH": "RAH", "AC_RAH": "RAH",
    "SAT": "SAT", "AC_SAT": "SAT", "PC_SAT": "SAT", "OAU_SAT": "SAT",

    # 제습 관련
    "DH_TEMP": "DH_TEMP",
    "DH_DEH": "DH_DEH",

    # 팬/모터 상태
    "SFST": "SFST", "SFST1": "SFST1", "SFST2": "SFST2",
    "PC_SFST": "PC_SFST", "AC_SFST": "AC_SFST", "OAU_SFST": "OAU_SFST",
    "EFST": "EFST", "DH_EFST": "DH_EFST",
    "RFST": "RFST",
    "DH_GMST": "DH_GMST",

    # CDU / COMP / HEATER
    "CDUST": "CDU", "OAU_CDUST": "CDU", "PC_CDUST": "CDU", "PC_CDU2ST": "CDU",
    "CDUSS": "CDU",
    "COMPST": "COMP", "COMPS": "COMP",
    "COMPST1": "COMPSS1", "COMPST2": "COMPSS2",
    "EHST": "EH", "EHST1": "EHSS1", "EHST2": "EHSS2", "EHST3": "EHSS3", "EHST4": "EHSS4",
    "HTST": "HTSS", "OAU_HTST1": "HTSS",

    # 외기
    "OA_TEMP": "OA_T", "OA_T": "OA_T", "OAT": "OA_T",
    "OA_HUMIDITY": "OA_H", "OA_H": "OA_H", "OAH": "OA_H",
}

# ── 모터/전기 부하 계열 (전기 kWh/비용 계산 대상) ──
MOTOR_TAGS = {
    "SFST","SFST1","SFST2","PC_SFST","AC_SFST","OAU_SFST",
    "EFST","DH_EFST",
    "RFST",
    "DH_GMST",
    "CDU","CDUSS",
    "COMP","COMPSS1","COMPSS2",
    "EH","EHSS1","EHSS2","EHSS3","EHSS4","HTSS",
}

# ── 센서 계열 (비용 X, 트렌드/밴드만) ──
SENSOR_TAGS = {
    "RAT","RAH",
    "SAT",
    "DH_TEMP","DH_DEH",
    "OA_T","OA_H",
}
# ─────────────────────────────────────
# 태그 문자열 → 표준 태그로 변환
# ─────────────────────────────────────
def 표준화_태그(raw: str) -> str | None:
    """
    'AHU01_CCV.PresentValue' / 'AC_AHU01_RAT' / 'OA_TEMP.PresentValue' 등에서
    유효 태그만 떼서 TAG_MAP 기준으로 표준화.
    """
    if not isinstance(raw, str):
        return None

    t = raw.strip().upper()

    # AHU 계열: 'AHUxx_XXXX...' 에서 XXXX만 추출
    if "AHU" in t:
        # AHU 뒤쪽만
        t = t.split("AHU", 1)[1]
        # AHU01_XXXX...
        t = t.split(".", 1)[0]
        t = t.split("_", 1)[-1]  # 맨 앞 AHUxx_ 제거 → 태그
    else:
        # OA_TEMP.PresentValue 등
        t = t.split(".", 1)[0]

    return TAG_MAP.get(t, t)  # 매핑 없으면 원본 리턴 (필요시 나중에 필터링)


# ─────────────────────────────────────
# CSV → 표준 포맷
# ─────────────────────────────────────
@st.cache_data
def 파일_정리(file_name: str, raw_bytes: bytes) -> pd.DataFrame:
    """
    CSV 형식
      - 1열: 포인트명 (예: 'AHU01_CCV.PresentValue', 'OA_TEMP.PresentValue')
      - 2열: 일시     (YYYYMMDDHHMM)
      - 3열: 값
    반환:
      ['공조기','항목명','datetime','값']
    """

    # 인코딩 추정 후 디코딩
    enc = (chardet.detect(raw_bytes) or {}).get("encoding") or "utf-8"
    raw = raw_bytes.decode(enc, errors="replace")

    # 앞 3열만 읽기 (헤더 없음 가정)
    df = pd.read_csv(
        StringIO(raw),
        sep=",",
        header=None,
        usecols=[0, 1, 2],
        names=["포인트", "날짜", "값"],
        dtype=str,
        skip_blank_lines=True,
    )

    # 날짜 파싱 (YYYYMMDDHHMM → datetime)
    df["datetime"] = pd.to_datetime(
        df["날짜"].str.strip(),
        format="%Y%m%d%H%M",
        errors="coerce"
    )

    # 값 숫자 변환 (쉼표/공백 제거)
    df["값"] = (
        df["값"].astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )

    # ── 포인트에서 공조기/항목명 추출 ──
    def extract_fields(p: str):
        if not isinstance(p, str):
            return (None, None)
        s = p.strip().upper()

        # OA 포인트
        if s.startswith("OA_") or "OA_TEMP" in s or "OA_HUMID" in s:
            tag = 표준화_태그(s)
            return ("OA", tag)

        # AHU 번호
        m_ahu = re.search(r"(AHU\d+(?:H)?)", s)
        ahu = normalize_ahu_name(m_ahu.group(1)) if m_ahu else None


        # AHUxx_XXXX 에서 XXXX 추출
        m_tag = re.search(r"AHU\d+(?:H)?_([^.\s]+)", s)
        raw_tag = m_tag.group(1) if m_tag else None
        tag = 표준화_태그(raw_tag) if raw_tag else None

        return (ahu, tag)

    df["공조기"], df["항목명"] = zip(*df["포인트"].map(extract_fields))

    # 파일명 기반 보정 (공조기 / OA)
    if df["공조기"].isna().any():
        fname = (file_name or "").upper()
        m = re.search(r"(AHU\d+(?:H)?)", fname)
        if m:
            df.loc[df["공조기"].isna(), "공조기"] = normalize_ahu_name(m.group(1))
        if "OA_" in fname or "OUTDOOR" in fname:
            df.loc[df["공조기"].isna(), "공조기"] = "OA"

    # OA 태그 최종 정리
    oa_map = {
        "OA_TEMP": "OA_T",
        "OA_T": "OA_T",
        "OAT": "OA_T",
        "OA_HUMIDITY": "OA_H",
        "OA_H": "OA_H",
        "OAH": "OA_H",
    }
    is_oa = df["공조기"].eq("OA")
    df.loc[is_oa, "항목명"] = df.loc[is_oa, "항목명"].map(
        lambda x: oa_map.get(str(x).upper(), x)
    )

    # 최종 필터링 & 정렬
    out = (
        df.dropna(subset=["datetime", "값", "항목명", "공조기"])
          .sort_values("datetime")
          .reset_index(drop=True)
    )

    return out[["공조기", "항목명", "datetime", "값"]]

@st.cache_data
def 캐시된_비용_계산_v2(ahu, 항목명, df, 최대열량):
    d = df[(df["공조기"] == ahu) & (df["항목명"] == 항목명)].sort_values("datetime").copy()
    if d.empty:
        return pd.DataFrame(columns=["날짜","kWh"]), 0.0

    d["dt_h"] = d["datetime"].diff().dt.total_seconds() / 3600
    d = d[(d["dt_h"] > 0) & (d["dt_h"] <= 5)].copy()

    v1, v2 = d["값"].shift(1), d["값"]

    if 항목명 == "전력":
        # kWh = kW × h (사다리꼴 적분)
        d["kWh"] = ((v1 + v2) / 2.0) * d["dt_h"]
    else:
        if 최대열량 <= 0:
            return pd.DataFrame(columns=["날짜","kWh"]), 0.0
        # 기존 냉수/스팀 (개도율 기반)
        d["kWh"] = ((v1 + v2) / 2.0) * 최대열량 * d["dt_h"] / 100.0 / 860.0

    d["날짜"] = d["datetime"].dt.date
    일별 = d.groupby("날짜", as_index=False)["kWh"].sum()
    return 일별, float(d["kWh"].sum())


@st.cache_data
def 캐시된_비용_계산_건식지원(ahu, 항목명, df, 열량맵):

    일별1, 총1 = 캐시된_비용_계산_v2(ahu, 항목명, df, 열량맵.get(ahu, 0))
    # 보조 항목: 건식제습형만 해당
    보조항목 = None
    if 항목명 == "CCV" and ahu in 건식제습형_공조기:
        보조항목 = "PC_CCV"
    elif 항목명 == "HCV" and ahu in 건식제습형_공조기:
        보조항목 = "DH_HCV"

    if 보조항목:
        보조_열량 = 항목_열량맵핑[보조항목].get(ahu, 0)
        if 보조_열량 > 0:
            df_sub = df[(df["공조기"] == ahu) & (df["항목명"] == 보조항목)].copy()
            if not df_sub.empty:
                일별2, 총2 = 캐시된_비용_계산_v2(ahu, 보조항목, df_sub, 보조_열량)
                일별1 = pd.merge(일별1, 일별2, on="날짜", how="outer", suffixes=("", "_보조")).fillna(0)
                일별1["kWh"] = 일별1["kWh"] + 일별1["kWh_보조"]
                총1 += 총2

    return 일별1, 총1

def 보간_및_이상값_탐지_fast(df, 최대열량, 이상값탐지=True):
    df = df.copy()

    # 1. datetime 정렬
    df = df.sort_values("datetime").reset_index(drop=True)

    # 2. 긴 결측 구간 제거 (3일 이상 = 432개)
    df["is_na"] = df["값"].isna()
    df["group"] = (df["is_na"] != df["is_na"].shift()).cumsum()
    group_sizes = df.groupby("group")["is_na"].sum()
    long_gaps = group_sizes[group_sizes >= 432].index
    df.loc[df["group"].isin(long_gaps), "값"] = np.nan

    # 3. 앞뒤 연속 결측 제거
    while len(df) > 0 and pd.isna(df.iloc[0]["값"]):
        df = df.iloc[1:]
    while len(df) > 0 and pd.isna(df.iloc[-1]["값"]):
        df = df.iloc[:-1]

    # 4. 5시간 초과 구간 제거
    df["시간차"] = df["datetime"].diff().dt.total_seconds() / 60
    초과인덱스 = df[df["시간차"] > 300].index
    for idx in 초과인덱스:
        if idx - 1 in df.index and idx in df.index:
            df.loc[[idx - 1, idx], "값"] = np.nan

    # ✅ 보간 없음 (interpolate 제거됨)

    # 5. 열량 계산
    df["시간간격"] = df["datetime"].diff().dt.total_seconds() / 3600
    df = df[(df["시간간격"] > 0) & (df["시간간격"] <= 12)].copy()

    v1, v2 = df["값"].shift(1), df["값"]
    df["열량_kWh"] = ((v1 + v2) / 2) * 최대열량 * df["시간간격"] / 100 / 860

    if 이상값탐지:
        df.loc[df["열량_kWh"] > 300, "열량_kWh"] = np.nan

    return df


# ✅ 열량 계산 시 누락 구간 앞뒤 평균으로 보간
def 보간_열량계산(df, 항목명, 최대열량):
    df = df.sort_values("datetime")
    df["시간간격"] = df["datetime"].diff().dt.total_seconds() / 3600
    df = df[df["시간간격"] > 0].copy()
    시간열 = df["datetime"]
    값열 = df["값"]
    열량 = []
    for i in range(len(df)):
        if i == 0:
            열량.append(0)
        else:
            v1, v2 = 값열.iloc[i - 1], 값열.iloc[i]
            평균값 = (v1 + v2) / 2
            열량.append(평균값 * 최대열량 * df["시간간격"].iloc[i] / 100 / 860)
    return pd.Series(열량, index=df.index)


def ahu_replace_once(x, mapping):
    return mapping[x] if x in mapping else x


# 절기 분류 함수
def 절기_분류(dt):
    month = dt.month
    if month in [4, 5, 10, 11]:
        return "간절기"
    elif month in [6, 7, 8, 9]:
        return "혹서기"
    else:
        return "혹한기"



def get_최대열량(row):
    ahu = row["공조기"]
    항목 = row["항목명"]
    if 항목 == "CCV":
        return 냉수최대열량.get(ahu, 0)
    elif 항목 == "PC_CCV":
        return PC_CCV_열량.get(ahu, 0)
    elif 항목 == "HCV":
        return 증기최대열량.get(ahu, 0)
    elif 항목 == "DH_HCV":
        return DH_HCV_열량.get(ahu, 0)
    else:
        return 0  # 혹시 RAT/RAH 같은 항목일 경우


단가_딕셔너리 = {
    2022: {"냉수단가": 295, "증기단가": 52300, "전기단가": 119},
    2023: {"냉수단가": 299, "증기단가": 57500, "전기단가": 154},
    2024: {"냉수단가": 304, "증기단가": 61600, "전기단가": 168},
    2025: {"냉수단가": 307, "증기단가": 65000, "전기단가": 182}
}

def get_단가(연도):
    return 단가_딕셔너리.get(연도, {"냉수단가": 304, "증기단가": 65000, "전기단가": 180})

항목명_한글 = {
    "CCV": "냉수코일",
    "AC_CCV": "냉수코일",
    "PC_CCV": "프리쿨러 냉수코일",
    "HCV": "스팀코일",
    "AC_HCV": "스팀코일",
    "DH_HCV": "제습 스팀코일",
    "RAT": "환기온도",
    "RAH": "환기습도",
    "전력": "전기",   # ← 전력도 전기로 통합
    "전기": "전기"
}

__all__ = [
    "냉수최대열량","증기최대열량","PC_CCV_열량","DH_HCV_열량","항목_열량맵핑",
    "서플라이팬용량","프로세스팬용량","배기팬용량","기어모터용량","로터모터용량","CDU용량","HEATER용량",
    "get_motor_device_kwh","파일_정리",
    "캐시된_비용_계산_v2","캐시된_비용_계산_건식지원","보간_및_이상값_탐지_fast","보간_열량계산",
    "ahu_replace_once","절기_분류","get_최대열량",
    "단가_딕셔너리","get_단가","항목명_한글",
    "건식제습형_공조기","냉각제습형_공조기",
]
