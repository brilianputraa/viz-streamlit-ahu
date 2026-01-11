# ============================================================================
# Standard Library Imports
# ============================================================================
import os
import re
import glob
import time
import hashlib
import bcrypt
import threading
import itertools
from datetime import datetime, timedelta

# ============================================================================
# Third-party Imports
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI  # pip install openai>=1.40
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# ============================================================================
# [추가됨] LLM 클라이언트 초기화 (ChatGPT & Gemini 지원)
# Added: OpenAI와 Gemini LLM 클라이언트 지원
# ============================================================================

# OpenAI (ChatGPT) Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API")
gpt_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
GEMINI_MODEL = "gemini-2.0-flash-exp"

# 페이지 설정 (앱 전체에서 단 1번만!)
st.set_page_config(page_title="공조기 데이터 처리", layout="wide")
 
# app2.py (상단)
try:
    # 패키지로 실행될 때
    from .common import (절기_분류, ahu_replace_once, 항목명_한글,
                          냉수최대열량, 증기최대열량, PC_CCV_열량, DH_HCV_열량, 항목_열량맵핑,
                          get_최대열량, 단가_딕셔너리, get_단가, get_motor_device_kwh,
                          건식제습형_공조기, 냉각제습형_공조기)
    from .loader import (HISTORY_DIR, update_history_results, load_final_results, load_detail_results, scan_and_update, load_oa_results, load_oa_daily, load_ahu_detail, get_items_from_final)
    from .viz import (draw_season_year_line, draw_overlay_by_shifted_datetime,
                      show_공조기별_총비용_요약, show_항목별_소모비용, add_band, 평균선추가,
                      BAND_RANGES_RAT, BAND_RANGES_RAH)
except ImportError:
    # 파일로 직접 실행될 때
    from common import (절기_분류, ahu_replace_once, 항목명_한글,
                        냉수최대열량, 증기최대열량, PC_CCV_열량, DH_HCV_열량, 항목_열량맵핑,
                        get_최대열량, 단가_딕셔너리, get_단가, get_motor_device_kwh,
                        건식제습형_공조기, 냉각제습형_공조기)
    from loader import (HISTORY_DIR, update_history_results, load_final_results, load_detail_results, scan_and_update, load_oa_results, load_oa_daily, load_ahu_detail, get_items_from_final)
    from viz import (draw_season_year_line, draw_overlay_by_shifted_datetime,
                     show_공조기별_총비용_요약, show_항목별_소모비용, add_band, 평균선추가,
                     BAND_RANGES_RAT, BAND_RANGES_RAH)

try:
    from .app2_loader import load_parquet_data, load_final_results_from_dir
except ImportError:
    from app2_loader import load_parquet_data, load_final_results_from_dir

WATCH_DIR = HISTORY_DIR

# ============================================================================
# [추가됨] data_adapter 임포트 (Parquet/Database 통합 데이터 접근 레이어)
# Added: data_adapter 모듈을 통해 Parquet와 Database 모드 지원
# ============================================================================
try:
    from .data_adapter import (
        DataAccessMode,
        load_final_results as load_adapted_final_results,
        load_ahu_detail as load_adapted_ahu_detail,
        load_oa_data as load_adapted_oa_data,
        ensure_ahu_query_lib
    )
except ImportError:
    from data_adapter import (
        DataAccessMode,
        load_final_results as load_adapted_final_results,
        load_ahu_detail as load_adapted_ahu_detail,
        load_oa_data as load_adapted_oa_data,
        ensure_ahu_query_lib
    )

# [수정됨] DB 모드 로더 라우팅 + ahu_query_lib 자동 경로 탐색
# Modified: ahu-backend-server 경로 자동 감지 및 DB 모드에서 data_adapter 사용
def load_ahu_detail_by_mode(ahu_name: str, mode: DataAccessMode) -> pd.DataFrame:
    if mode == DataAccessMode.DATABASE:
        return load_adapted_ahu_detail(ahu_name, mode=mode)
    return load_ahu_detail(ahu_name)

# 파일 해시
def _list_csvs(folder: str):
    return sorted(glob.glob(os.path.join(folder, "*.csv")))

def _files_signature(paths):
    import hashlib
    md5 = hashlib.md5()
    for p in sorted(paths):
        try:
            with open(p, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            continue
        md5.update(p.encode("utf-8"))
        md5.update(hashlib.md5(data).digest())
    return md5.hexdigest()

# 이벤트 객체 (스레드-세이프)
@st.cache_resource
def get_reload_event():
    return threading.Event()

@st.cache_resource
def start_watcher(path: str, _ev):
    class _Handler(FileSystemEventHandler):
        def on_modified(self, event):
            # 디렉토리 이벤트는 무시, CSV 파일만 감지
            if not event.is_directory and event.src_path.endswith(".csv"):
                time.sleep(0.5)  # 저장 중간에 안 걸리게 딜레이
                _ev.set()

    observer = Observer()
    observer.schedule(_Handler(), path, recursive=False)
    observer.start()
    return observer

reload_event = get_reload_event()
_ = start_watcher(WATCH_DIR, reload_event)

# 해시 기반 보조 체크
if "files_sig" not in st.session_state:
    st.session_state["files_sig"] = _files_signature(_list_csvs(WATCH_DIR))
else:
    current_sig = _files_signature(_list_csvs(WATCH_DIR))
    if current_sig != st.session_state["files_sig"]:
        st.session_state["files_sig"] = current_sig
        reload_event.set()

# 🔥 메인 루프에서 이벤트 확인
if reload_event.is_set():
    reload_event.clear()
    new_sig = _files_signature(_list_csvs(WATCH_DIR))
    if new_sig != st.session_state["files_sig"]:
        st.session_state["files_sig"] = new_sig
        st.toast("📂 새 CSV 감지 → 자동 새로고침", icon="✅")
        st.rerun()

# 📂 데이터 로드
st.header("📂 데이터 로드")

# --- 데이터 로딩 로직 개선 ---
progress_bar = st.progress(0, text="파일 분석 준비 중...")
def update_progress(current, total, file_name):
    progress_bar.progress(current / total, text=f"📂 파일 분석 중... ({current}/{total}) - {file_name}")

# ============================================================================
# [수정됨] 데이터 소스 선택 (Parquet/Database 모드 직접 선택)
# Original: Parquet 파일만 직접 로드
# Modified: 사용자가 Parquet 또는 Database 모드를 직접 선택
# ============================================================================

# First-time mode selection (shown before data loading)
if "data_source_mode" not in st.session_state:
    st.markdown("---")
    st.markdown("### 🎯 데이터 소스 선택")
    st.markdown("분석할 데이터 소스를 선택해주세요.")

    # Modern card-style selection
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; border: 2px solid #cbd5e0;">
            <h3 style="color: #2d3748; margin-bottom: 10px;">📁 Parquet Files</h3>
            <p style="color: #718096; font-size: 14px; margin-bottom: 15px;">
                로컬 Parquet 파일에서 데이터 로드
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Parquet 모드 선택", key="select_parquet", use_container_width=True, type="secondary"):
            st.session_state["data_source_mode"] = "parquet"
            st.rerun()

    with col2:
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px; background-color: #ebf8ff; border: 2px solid #4299e1;">
            <h3 style="color: #2c5282; margin-bottom: 10px;">🗄️ Database</h3>
            <p style="color: #4a5568; font-size: 14px; margin-bottom: 15px;">
                PostgreSQL 데이터베이스에서 실시간 데이터 로드
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Check if ahu_query_lib is available
        db_available = ensure_ahu_query_lib() is not None

        if st.button("Database 모드 선택", key="select_database", use_container_width=True, type="primary" if db_available else "secondary"):
            if db_available:
                st.session_state["data_source_mode"] = "database"
                st.rerun()
            else:
                st.error("❌ ahu_query_lib가 설치되지 않았습니다.")
                st.code("export PYTHONPATH=/path/to/ahu-backend-server:$PYTHONPATH", language="bash")

    st.markdown("---")
    st.stop()

# Load data based on selected mode
selected_mode = st.session_state.get("data_source_mode", "parquet")

if selected_mode == "parquet":
    should_update = "데이터로드완료" not in st.session_state or reload_event.is_set()
    with st.spinner("📂 CSV → parquet 업데이트 중..." if should_update else "📂 Parquet 데이터 로드 중..."):
        df_final_all, df_oa_daily, df_oa_all, did_update = load_parquet_data(
            should_update=should_update,
            update_fn=lambda: update_history_results(progress_callback=update_progress),
            final_fn=load_final_results,
            oa_daily_fn=load_oa_daily,
            oa_all_fn=load_oa_results,
        )

    if did_update:
        st.session_state["데이터로드완료"] = True
        progress_bar.empty()
        st.success(f"✅ 집계 데이터 {len(df_final_all)}건, OA(일평균) {len(df_oa_daily)}건, OA(고해상도) {len(df_oa_all)}건 로드 완료")
    else:
        st.success("✅ 기존 데이터 사용")

    st.session_state["initial_data_loaded"] = True
    st.session_state["data_source_used"] = "parquet"
else:
    if "initial_data_loaded" not in st.session_state or st.session_state.get("data_source_used") != "database":
        with st.spinner("🔄 DATABASE 모드로 데이터 로드 중..."):
            try:
                # Load from Database
                df_final_all = load_adapted_final_results(mode=DataAccessMode.DATABASE)
                df_oa_daily = load_adapted_oa_data(mode=DataAccessMode.DATABASE, daily=True)
                df_oa_all = load_adapted_oa_data(mode=DataAccessMode.DATABASE, daily=False)

                # [수정됨] None 반환 대비 (ahu_query_lib에서 None 리턴 시 오류 방지)
                # Modified: None -> empty DataFrame 변환
                if df_final_all is None:
                    df_final_all = pd.DataFrame()
                if df_oa_daily is None:
                    df_oa_daily = pd.DataFrame()
                if df_oa_all is None:
                    df_oa_all = pd.DataFrame()

                st.session_state["initial_data_loaded"] = True
                st.session_state["data_source_used"] = "database"
                st.success("✅ 데이터 로드 완료 (DATABASE 모드)")

            except Exception as e:
                st.error(f"❌ 데이터 로드 실패: {e}")
                import traceback
                st.error(traceback.format_exc())
                st.info("💡 다른 모드를 선택하려면 세션을 다시 시작하세요.")
                st.stop()
    else:
        st.success("✅ 세션 데이터 사용 (DATABASE 모드)")
        df_final_all = load_adapted_final_results(mode=DataAccessMode.DATABASE)
        df_oa_daily = load_adapted_oa_data(mode=DataAccessMode.DATABASE, daily=True)
        df_oa_all = load_adapted_oa_data(mode=DataAccessMode.DATABASE, daily=False)

if selected_mode == "parquet" and (df_final_all is None or df_final_all.empty):
    st.error("데이터를 불러오지 못했습니다. history 경로/날짜를 확인하세요.")
    st.stop()

if selected_mode == "parquet":
    FINAL_DIR = os.getenv("AHU_FINAL_DIR", r"C:\Users\User\Desktop\ahu_app_results\final_results")
    override_df = load_final_results_from_dir(FINAL_DIR)
    if not override_df.empty:
        df_final_all = override_df
    
#====================================================================================
# 로그인 기능, 자동 rerun 기능 등 기타 코드... (이 부분은 변경하지 않음)
#====================================================================================

st.title("📊 공조기 분석 시스템")

# ============================================================================
# [추가됨] 데이터 소스 선택 (Database vs Parquet)
# Added: 사이드바에서 Parquet/Database 모드 선택 기능
# ============================================================================
st.sidebar.markdown("---")

if st.sidebar.button("🧹 Parquet 강제 재분석", key="force_rebuild_parquet"):
    st.session_state["데이터로드완료"] = False
    st.session_state["initial_data_loaded"] = False
    reload_event.set()
    with st.spinner("Parquet 검증을 위해 다시 생성 중..."):
        update_history_results(progress_callback=update_progress)
    st.sidebar.success("✅ Parquet 재분석 완료")
    st.experimental_rerun()

# Get current data source from session state
current_data_source = st.session_state.get("data_source_used", "parquet")

# Set default index based on auto-detected source
default_index = 1 if current_data_source == "database" else 0

data_source_mode = st.sidebar.radio(
    "🗄️ 데이터 소스",
    options=["Parquet Files", "Database"],
    index=default_index,
    help=f"현재: {current_data_source.upper()} 모드 (Parquet Files 또는 Database 선택)"
)

# Convert to DataAccessMode enum
mode = DataAccessMode.PARQUET if data_source_mode == "Parquet Files" else DataAccessMode.DATABASE

# Display current mode status
if mode == DataAccessMode.DATABASE:
    try:
        aql = ensure_ahu_query_lib()
        if not aql:
            raise ImportError("ahu_query_lib not available")
        if current_data_source == "database":
            st.sidebar.success("✅ Database mode (auto-detected)")
        else:
            st.sidebar.success("✅ Database connected")
        st.sidebar.caption(f"ahu_query_lib v{aql.__version__}")
    except ImportError:
        st.sidebar.error("❌ ahu_query_lib not installed")
        st.sidebar.caption("Run: export PYTHONPATH=/path/to/ahu-backend-server:$PYTHONPATH")
        # Fallback to parquet mode if library not available
        mode = DataAccessMode.PARQUET
else:
    if current_data_source == "parquet":
        st.sidebar.info("📁 Using Parquet files (auto-detected)")
    else:
        st.sidebar.info("📁 Using Parquet files")

st.sidebar.markdown("---")

# ============================================================================
# [수정됨] 데이터 소스에 따른 데이터 로드 (Parquet/Database 모드 지원)
# Modified: data_adapter를 통해 선택된 모드로 데이터 로드
# ============================================================================
# Note: Database mode의 경우 energy 데이터는 비어있을 수 있습니다
# (energy_readings 테이블이 비어있음). Sensor 데이터는 정상 작동합니다.
if mode == DataAccessMode.DATABASE and st.sidebar.button("🔄 DB에서 데이터 다시 로드", key="reload_db_data"):
    with st.spinner("데이터베이스에서 데이터 로드 중..."):
        try:
            # Load data using data_adapter
            df_final_all = load_adapted_final_results(mode=mode)
            외기df_daily = load_adapted_oa_data(mode=mode, daily=True)
            외기df_hourly = load_adapted_oa_data(mode=mode, daily=False)

            all_df = df_final_all.copy()

            # [수정됨] Empty DataFrame 체크 추가
            # Normalize AHU names (only if DataFrame has the column)
            if not all_df.empty and "공조기" in all_df.columns:
                all_df["공조기"] = (
                    all_df["공조기"]
                      .astype(str)
                      .str.replace(r"AHU-?(\d+)(H)?", lambda m: f"AHU{int(m.group(1)):02d}" + (m.group(2) or ""), regex=True)
                )
            if not df_final_all.empty and "공조기" in df_final_all.columns:
                df_final_all["공조기"] = (
                    df_final_all["공조기"]
                      .astype(str)
                      .str.replace(r"AHU-?(\d+)(H)?", lambda m: f"AHU{int(m.group(1)):02d}" + (m.group(2) or ""), regex=True)
                )

            # [수정됨] None 반환 대비 (ahu_query_lib에서 None 리턴 시 오류 방지)
            # Modified: None -> empty DataFrame 변환
            if df_final_all is None:
                df_final_all = pd.DataFrame()
            if 외기df_daily is None:
                외기df_daily = pd.DataFrame()
            if 외기df_hourly is None:
                외기df_hourly = pd.DataFrame()

            st.success(f"✅ DB 데이터 로드 완료: {len(df_final_all)}건 (energy), {len(외기df_daily)}건 (OA daily), {len(외기df_hourly)}건 (OA hourly)")
            st.sidebar.success("✅ Database data loaded")

            if df_final_all.empty:
                st.warning("⚠️ Energy 데이터가 비어있습니다. energy_readings 테이블에 데이터가 없습니다.")
                st.info("💡 Sensor 데이터 (Detail view)는 정상 작동합니다. Energy 데이터는 ETL이 필요합니다.")

        except Exception as e:
            st.error(f"❌ DB 데이터 로드 실패: {e}")
            import traceback
            st.error(traceback.format_exc())

# 여기서부터는 all_df와 외기df를 사용합니다.
all_df = df_final_all.copy()
외기df_daily  = df_oa_daily.copy()
외기df_hourly = df_oa_all.copy()

# [수정됨] Database mode can return timestamp/date columns instead of datetime; normalize once here.
def _normalize_datetime_column(df: pd.DataFrame) -> None:
    if df is None or df.empty or "datetime" in df.columns:
        return
    candidate_cols = ["timestamp", "date", "날짜"]
    for col in candidate_cols:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            if getattr(dt.dt, "tz", None) is not None:
                dt = dt.dt.tz_localize(None)
            df["datetime"] = dt
            return

_normalize_datetime_column(all_df)

# all_df = df_final_all.copy() 바로 아래에 추가
# "AHU-07", "AHU7", "AHU007", "AHU07H" 등 변형을 모두 "AHU07" 또는 "AHU07H"로 정규화
# [수정됨] Empty DataFrame 체크 추가 (Database mode에서 energy 데이터가 비어있을 경우 대응)
if not all_df.empty and "공조기" in all_df.columns:
    all_df["공조기"] = (
        all_df["공조기"]
          .astype(str)
          .str.replace(r"AHU-?(\d+)(H)?", lambda m: f"AHU{int(m.group(1)):02d}" + (m.group(2) or ""), regex=True)
    )

# 요약 스냅샷도 동일 키로 맞추기 (df_final_all도 같은 규칙으로)
if not df_final_all.empty and "공조기" in df_final_all.columns:
    df_final_all["공조기"] = (
        df_final_all["공조기"]
          .astype(str)
          .str.replace(r"AHU-?(\d+)(H)?", lambda m: f"AHU{int(m.group(1)):02d}" + (m.group(2) or ""), regex=True)
    )


# final_analysis.parquet으로부터 데이터 추출
def get_daily_kwh(df: pd.DataFrame, ahu: str, item: str) -> pd.DataFrame:
    """
    parquet 기반: 공조기별, 항목별 일별 kWh 합산
    """
    tmp = df[(df["공조기"] == ahu) & (df["항목명"] == item)].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["날짜", "kWh"])

    # 날짜 컬럼 생성
    if "날짜" in tmp.columns:
        tmp["날짜"] = pd.to_datetime(tmp["날짜"])
    else:
        tmp["날짜"] = pd.to_datetime(tmp["datetime"]).dt.normalize()

    # 일별 kWh 합산
    out = tmp.groupby("날짜", as_index=False)["kWh"].sum()

    return out


def make_top_summary(base_df: pd.DataFrame, raw: bool = False) -> pd.DataFrame:
    """
    공조기별 전력/냉수코일/스팀코일/총비용 집계.

    우선순위:
      - loader.py에서 만들어 준 표준 컬럼 사용:
        냉수_비용(원), 스팀_비용(원), 전력_비용(원), 총합_비용(원)
      - 없으면 개별 코일/모터 비용 컬럼을 이용해 재계산
      - 총비용(원)은 항상 전력 + 냉수 + 스팀의 합으로 다시 계산 (총합_비용(원) 맹신 X)
    """
    if base_df is None or base_df.empty:
        return pd.DataFrame(
            columns=["공조기", "총비용(원)", "전력사용량(원)", "냉수코일비용(원)", "스팀코일비용(원)"]
        )

    df = base_df.copy()

    # 숫자화
    for c in df.columns:
        if "비용(원)" in str(c):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── 냉수 코일 비용 ──
    if "냉수_비용(원)" in df.columns:
        df["냉수코일비용(원)"] = df["냉수_비용(원)"]
    elif "비용(원)_냉수" in df.columns:
        df["냉수코일비용(원)"] = df["비용(원)_냉수"]
    else:
        냉수_cols = [c for c in df.columns if c in ["비용(원)_CCV", "비용(원)_PC_CCV", "비용(원)_AC_CCV"]]
        df["냉수코일비용(원)"] = df[냉수_cols].sum(axis=1, min_count=1) if 냉수_cols else np.nan

    # ── 스팀 코일 비용 ──
    if "스팀_비용(원)" in df.columns:
        df["스팀코일비용(원)"] = df["스팀_비용(원)"]
    elif "비용(원)_스팀" in df.columns:
        df["스팀코일비용(원)"] = df["비용(원)_스팀"]
    else:
        스팀_cols = [c for c in df.columns if c in ["비용(원)_HCV", "비용(원)_DH_HCV", "비용(원)_AC_HCV"]]
        df["스팀코일비용(원)"] = df[스팀_cols].sum(axis=1, min_count=1) if 스팀_cols else np.nan

    # ── 전력 비용 ──
    if "전력_비용(원)" in df.columns:
        df["전력사용량(원)"] = df["전력_비용(원)"]
    elif "비용(원)_전력" in df.columns:
        df["전력사용량(원)"] = df["비용(원)_전력"]
    else:
        # 코일/총합/이미 계산된 표준 컬럼 제외한 나머지 비용 → 전력으로 간주
        exclude = {
            "냉수코일비용(원)", "스팀코일비용(원)",
            "냉수_비용(원)", "스팀_비용(원)",
            "비용(원)_냉수", "비용(원)_스팀",
            "총합_비용(원)", "총합_비용"
        }
        # CCV/HCV 계열(이미 냉수/스팀에 포함해야 하는 것)도 제외
        coil_patterns = ("_CCV", "_PC_CCV", "_AC_CCV", "_HCV", "_DH_HCV", "_AC_HCV")

        power_cols = []
        for c in df.columns:
            if not str(c).endswith("비용(원)"):
                continue
            if c in exclude:
                continue
            if any(pat in c for pat in coil_patterns):
                continue
            power_cols.append(c)

        df["전력사용량(원)"] = df[power_cols].sum(axis=1, min_count=1) if power_cols else np.nan

    # ── 총비용: 항상 전력 + 냉수 + 스팀로 재계산 ──
    base_cols = [c for c in ["전력사용량(원)", "냉수코일비용(원)", "스팀코일비용(원)"] if c in df.columns]
    if base_cols:
        df["총비용(원)"] = df[base_cols].sum(axis=1, min_count=1)
    else:
        # 최후의 보루: 모든 비용(원) 합 (총합/표준컬럼 제외)
        tmp_cols = [
            c for c in df.columns
            if str(c).endswith("비용(원)")
            and not str(c).startswith("총합_")
        ]
        df["총비용(원)"] = df[tmp_cols].sum(axis=1, min_count=1) if tmp_cols else np.nan

    # ── 공조기별 집계 ──
    summary = (
        df.groupby("공조기", as_index=False)[
            ["총비용(원)", "전력사용량(원)", "냉수코일비용(원)", "스팀코일비용(원)"]
        ]
        .sum(min_count=1)
    )

    if raw:
        return summary

    # ── 표시용 포맷 ──
    fmt = summary.copy()
    num_cols = [c for c in fmt.columns if c != "공조기"]
    for c in num_cols:
        fmt[c] = fmt[c].apply(
            lambda x: f"{int(round(x)):,}" if pd.notna(x) else ""
        )
    fmt.index = np.arange(1, len(fmt) + 1)
    fmt.index.name = "No"
    return fmt


# ✅ 세션 업데이트
st.session_state['uploaded_df'] = all_df

# ============================================================================
# [수정됨] Empty DataFrame 체크 추가 (Database mode에서 energy 데이터가 비어있을 경우 대응)
# Modified: Database mode에서 energy 데이터가 비어있을 경우 처리 건너뛰기
# ============================================================================
# Energy 데이터가 비어있으면 처리 건너뛰기 (Database mode ETL 필요)
if all_df.empty or "datetime" not in all_df.columns:
    st.warning("⚠️ Energy 데이터가 비어있습니다. energy_readings 테이블에 데이터가 없습니다.")
    st.info("💡 Sensor 데이터 (Detail view)는 정상 작동합니다. Energy 데이터는 ETL이 필요합니다.")
    st.info("💡 데이터를 확인하려면 아래로 스크롤하세요.")
else:
    # Energy 데이터가 있으면 연도/절기 컬럼 추가
    all_df["연도"] = all_df["datetime"].dt.year
    all_df["절기"] = all_df["datetime"].apply(절기_분류)

    # Energy 데이터가 있을 때만 아래 처리 실행
    ENERGY_DATA_AVAILABLE = True


# 1) 코일(냉수/스팀) 별칭 정의: 두 번째 표처럼 AC_/PC_/DH_가 붙어도 같은 코일로 취급
COIL_ALIASES = {
    "CCV": ["CCV", "AC_CCV", "PC_CCV"],      # 냉수 코일
    "HCV": ["HCV", "AC_HCV", "DH_HCV"],      # 스팀 코일
}

# 2) 접두/접미 패턴으로 컬럼 찾기 유틸
def _find_metric_cols(df, alias_list, metric): 
    # metric: "kWh" 또는 "비용(원)"
    cols = []
    for alias in alias_list:
        # 접두사형: kWh_ALIAS, 비용(원)_ALIAS
        cols += [c for c in df.columns if c == f"{metric}_{alias}"]
        # 접미사형: ALIAS_kWh, ALIAS_비용(원)  (대소문자 kWh 모두 허용)
        if metric == "kWh":
            cols += [c for c in df.columns if c.lower() == f"{alias}_kwh".lower()]
        else:
            cols += [c for c in df.columns if c == f"{alias}_비용(원)"]
    # 중복 제거 (있을 수 있음)
    return sorted(list(dict.fromkeys(cols)))

# 3) 코일별(kWh/비용) 정규화 컬럼 생성: 여러 별칭을 합쳐서 하나로 만들기
for coil, aliases in COIL_ALIASES.items():
    kwh_cols  = _find_metric_cols(all_df, aliases, "kWh")
    cost_cols = _find_metric_cols(all_df, aliases, "비용(원)")
    if kwh_cols:
        all_df[kwh_cols] = all_df[kwh_cols].apply(pd.to_numeric, errors="coerce")
        all_df[f"kWh_{coil}"] = all_df[kwh_cols].sum(axis=1, min_count=1)
    if cost_cols:
        all_df[cost_cols] = all_df[cost_cols].apply(pd.to_numeric, errors="coerce")
        all_df[f"비용(원)_{coil}"] = all_df[cost_cols].sum(axis=1, min_count=1)

# 4) 전력(모터류) 비용: 코일/총합/전력/냉수/스팀은 제외하고, 접두/접미 둘 다 인식
EXCLUDE_TOKENS = set().union(*COIL_ALIASES.values()) | {"냉수","스팀","총합","전력"}

def _is_power_cost(col):
    # 비용(원)_장치
    if col.startswith("비용(원)_"):
        dev = col.split("_", 1)[1]
        return dev not in EXCLUDE_TOKENS
    # 장치_비용(원)
    if col.endswith("_비용(원)"):
        dev = col.rsplit("_", 1)[0]
        return dev not in EXCLUDE_TOKENS
    return False

power_cost_cols = [c for c in all_df.columns if _is_power_cost(c)]
calc_power_cost = None
if power_cost_cols:
    all_df[power_cost_cols] = all_df[power_cost_cols].apply(pd.to_numeric, errors="coerce")
    calc_power_cost = all_df[power_cost_cols].sum(axis=1, min_count=1)

# 5) 비용 소스가 없거나 전부 NaN이면 → 전기 kWh 합 × 연도별 단가로 보정 (kWh 접두/접미 + 소문자 허용)
def _is_power_kwh(col):
    if col.startswith(("kWh_","kwh_")):
        dev = col.split("_", 1)[1]
        return dev not in {"냉수","스팀"} and dev not in EXCLUDE_TOKENS
    if col.lower().endswith("_kwh"):
        dev = col.rsplit("_", 1)[0]
        return dev not in {"냉수","스팀"} and dev not in EXCLUDE_TOKENS
    return False

if (calc_power_cost is None) or calc_power_cost.isna().all():
    kwh_cols = [c for c in all_df.columns if _is_power_kwh(c)]
    if kwh_cols:
        all_df[kwh_cols] = all_df[kwh_cols].apply(pd.to_numeric, errors="coerce")
        kwh_sum = all_df[kwh_cols].sum(axis=1, min_count=1)
        def _price_by_year(y):
            try:
                return 단가_딕셔너리[int(y)]["전력(원/kWh)"]
            except Exception:
                return np.nan
        price = all_df["datetime"].dt.year.map(_price_by_year)
        calc_power_cost = kwh_sum * price

# 6) 전력_비용(원) 결측만 보강 (기존값 있으면 존중)
if calc_power_cost is not None:
    # calc_power_cost를 all_df 길이에 맞는 시리즈로 보장
    calc_power_cost = pd.Series(calc_power_cost, index=all_df.index)
    
    if "전력_비용(원)" in all_df.columns:
        s = pd.to_numeric(all_df["전력_비용(원)"], errors="coerce")
        mask = s.isna()
        # 인덱스 정렬/재색인 없이 "같은 위치"만 채움
        s.loc[mask] = calc_power_cost.loc[mask].values
        all_df["전력_비용(원)"] = s
    else:
        all_df["전력_비용(원)"] = calc_power_cost.values


# ============================================================================
# [수정됨] Energy 데이터 처리 (Empty DataFrame 체크 추가)
# Modified: Energy 데이터가 비어있을 경우 처리 건너뛰기
# ============================================================================

# Energy 데이터가 있을 때만 처리
if 'ENERGY_DATA_AVAILABLE' in locals() and ENERGY_DATA_AVAILABLE:
    # --- 날짜 범위 선택 (all_df 로드 후 바로) ---
    start_date = all_df["datetime"].min().date()
    end_date   = all_df["datetime"].max().date()

    전체날짜범위 = st.date_input(
        "📅 분석할 날짜 범위 선택",
        (start_date, end_date),
        key="전체날짜"
    )
    if isinstance(전체날짜범위, (tuple, list)):
        if len(전체날짜범위) >= 2:
            시작 = pd.Timestamp(전체날짜범위[0])
            종료 = pd.Timestamp(전체날짜범위[1]) + pd.Timedelta(days=1)
        elif len(전체날짜범위) == 1:
            시작 = pd.Timestamp(전체날짜범위[0])
            종료 = 시작 + pd.Timedelta(days=1)
        else:
            시작 = pd.Timestamp(start_date)
            종료 = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    else:
        시작 = pd.Timestamp(전체날짜범위)
        종료 = 시작 + pd.Timedelta(days=1)

    # 이 구간에 맞게 필터링된 df
    all_df_range = all_df[
        (all_df["datetime"] >= 시작) &
        (all_df["datetime"] <  종료)
    ].copy()

    # --- 공조기별 요약 (이제 기간 적용) ---
    top_summary_df = make_top_summary(all_df_range)
    st.markdown("### 📊 공조기별 에너지 사용량/비용 요약")
    st.dataframe(top_summary_df, use_container_width=True)

    공조기목록 = sorted(
        set(df_final_all["공조기"].dropna().unique())
        | set(all_df["공조기"].dropna().unique())
    )
    항목목록 = get_items_from_final(all_df)
else:
    # Energy 데이터가 없을 때의 기본값 설정
    st.info("💡 Energy 데이터가 없습니다. Sensor 데이터만 사용 가능합니다.")
    공조기목록 = sorted(set(df_final_all["공조기"].dropna().unique())) if not df_final_all.empty and "공조기" in df_final_all.columns else []
    항목목록 = []
    시작 = pd.to_datetime('2025-01-01')
    종료 = pd.to_datetime('2025-12-31')

AHU_RAT_LIMITS = {
    "AHU01": [17.9, 25.1], "AHU02": [17.9, 25.1], "AHU03": [17.9, 25.1], "AHU04": [17.9, 25.1], "AHU05": [17.9, 25.1],
    "AHU06": [17.9, 25.1], "AHU07": [17.9, 25.1], "AHU08": [17.9, 25.1], "AHU09": [17.9, 25.1], "AHU10": [17.9, 25.1],
    "AHU11": [17.9, 25.1], "AHU12": [17.9, 25.1], "AHU13": [17.9, 22.1], "AHU14": [17.9, 25.1], 
    "AHU020": [0.9, 25.1], "AHU021": [0.9, 30.1], "AHU022": [17.9, 25.1], "AHU023": [17.9, 25.1],
    "AHU024": [18, 22], "AHU025": [18, 22], "AHU026": [17.9, 25.1], "AHU27": [17.9, 25.1], 
    "AHU39": [14.9, 25.1], "AHU45": [18, 22]
}

AHU_RAH_LIMITS = {
    "AHU01": [75.1], "AHU02": [75.1], "AHU03": [75.1], "AHU05": [75.1],
    "AHU06": [75.1], "AHU07": [75.1], "AHU09": [75.1], "AHU10": [75.1],
    "AHU11": [75.1], "AHU13": [65.1], "AHU14": [75.1],
    "AHU020": [70.1], "AHU021": [70.1], "AHU022": [75.1],
    "AHU024": [70.1], "AHU025": [70.1], "AHU026": [75.1], "AHU27": [75.1],
    "AHU39": [75.1], "AHU45": [70.1]
}  

# ============================================================================
# [수정됨] Empty DataFrame 체크 추가 (연도 목록 추출)
# Modified: Empty DataFrame일 때 기본값 반환
# ============================================================================
연도목록 = sorted(all_df["datetime"].dt.year.unique()) if not all_df.empty and "datetime" in all_df.columns else []







def _format_number(v, unit="원"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    try:
        return f"{int(round(float(v))):,}{unit if unit else ''}"
    except Exception:
        return str(v)

def build_gpt_context(ahu, start_dt, end_dt, daily_df, baseline_df, k_sigma, highlight_year, topN_df):
    """
    현재 화면의 핵심 값만 압축해서 텍스트 컨텍스트로 만들기.
    - 일수/평균/초과합계 등
    """
    if daily_df is None or daily_df.empty:
        return f"공조기 {ahu} / {start_dt.date()}~{(end_dt - pd.Timedelta(days=1)).date()} 기간에 데이터 없음."

    # 기간 요약
    days = daily_df["날짜"].nunique()
    total_cost = daily_df["총비용(원)"].sum()
    exceed_cost = daily_df["초과비용(원)"].sum()
    avg_cost = daily_df["총비용(원)"].mean()

    # 연도별 합계(간단)
    by_year = (
        daily_df.groupby("연도")["총비용(원)"]
        .sum()
        .sort_index()
        .to_dict()
    )

    # 기준선 요약
    baseline_mean = baseline_df["기준선(원)"].mean() if not baseline_df.empty else np.nan

    # TopN 표 간단 정리
    top_lines = []
    if topN_df is not None and not topN_df.empty:
        for _, r in topN_df.head(5).iterrows():
            top_lines.append(
                f"- {r['날짜'].date()} | 비용 {_format_number(r['총비용(원)'])} vs 기준선 {_format_number(r['기준선(원)'])} → 초과 {_format_number(r['초과비용(원)'])}"
            )

    ctx = []
    ctx.append(f"[기본정보] 공조기: {ahu}, 기간: {start_dt.date()} ~ {(end_dt - pd.Timedelta(days=1)).date()} (일수: {days}일)")
    ctx.append(f"[기준설정] 기준 = 월-일 평균 + {k_sigma:.1f}σ, 하이라이트 연도 = {highlight_year}")
    ctx.append(f"[총괄] 총비용: {_format_number(total_cost)}, 일평균: {_format_number(avg_cost)}, 기준선 평균: {_format_number(baseline_mean)}, 잠재 절감(초과합): {_format_number(exceed_cost)}")
    ctx.append("[연도별 총비용] " + ", ".join([f"{y}: {_format_number(v)}" for y, v in by_year.items()]))

    if top_lines:
        ctx.append("[초과 Top5]")
        ctx.extend(top_lines)

    return "\n".join(ctx)



탭2, 탭3, 탭5 = st.tabs(["⚡ 에너지 사용량/소모비용 분석", "공조기 간 에너지 분석", "📊 항목 분석"])

with 탭2:

    # ============================================================================
    # [수정됨] Energy 데이터 체크 (탭 시작 부분)
    # Modified: Energy 데이터가 없으면 메시지 표시 후 탭 종료
    # ============================================================================
    if not ('ENERGY_DATA_AVAILABLE' in locals() and ENERGY_DATA_AVAILABLE):
        st.warning("⚠️ **Energy 데이터가 없습니다**")
        st.info("""
        💡 **Energy 데이터를 사용하려면:**

        1. **Parquet 모드**: `history` 폴더에 파quet 파일이 있는지 확인하세요

        2. **Database 모드**: Energy 데이터 자동 로드
           - **Airflow Webserver** 실행 중인지 확인하세요
           - Smart file monitoring이 자동으로 `ahu_readings_staging` → `energy_readings` ETL 실행
           - Airflow DAG: `etl_sensor_to_energy`가 주기적으로 실행됩니다

        🔧 **Airflow 상태 확인:**
        - Webserver: `http://localhost:8080`
        - CLI: `airflow dags list`
        - Logs: `airflow logs etl_sensor_to_energy --last 1`

        🔧 **수동 ETL (Airflow가 실행되지 않을 때):**
        ```sql
        -- ahu_monitoring DB에서 실행
        INSERT INTO ahu_data.energy_readings (timestamp, ahu_id, metric_name, value, unit)
        SELECT
            timestamp,
            ahu_id,
            CASE
                WHEN 항목명 IN ('CCV', 'PC_CCV') THEN 'ccv_cold_water_kwh'
                WHEN 항목명 IN ('HCV', 'DH_HCV') THEN 'hcv_steam_kwh'
                WHEN 항목명 = 'SFST' THEN 'ac_sf_electricity_kwh'
                ELSE 'other'
            END as metric_name,
            SUM(값) as value,
            'kWh' as unit
        FROM ahu_data.ahu_readings_staging
        WHERE 항목명 IN ('CCV', 'PC_CCV', 'HCV', 'DH_HCV', 'SFST')
        GROUP BY timestamp, ahu_id,
                 CASE WHEN 항목명 IN ('CCV', 'PC_CCV') THEN 'ccv_cold_water_kwh'
                      WHEN 항목명 IN ('HCV', 'DH_HCV') THEN 'hcv_steam_kwh'
                      WHEN 항목명 = 'SFST' THEN 'ac_sf_electricity_kwh'
                      ELSE 'other' END;
        ```
        """)
        st.success("✅ **Sensor 데이터는 정상 작동합니다!**")
        st.info("💡 Sensor 데이터 분석은 다른 탭을 이용해주세요.")
        st.stop()

    # ─────────────────────────────────────────────────────────
    # 🤖 GPT 인사이트 패널 (탭2의 맨 위로 이동!)
    with st.sidebar:
        st.header("🧠 ChatGPT 인사이트")
        st.caption(f"모델: `{GPT_MODEL}`")

        # 입력칸/버튼은 항상 렌더링, 키 없으면 비활성화
        user_q = st.text_area(
            "질문 입력",
            placeholder="예: 여름철 냉수비용 급등 원인은?",
            height=80,
            key="gpt_q",
            disabled=(gpt_client is None),
        )
        col1, col2 = st.columns(2)
        gen_clicked = col1.button("🔍 GPT 인사이트 생성", use_container_width=True,
                                key="gen_insight_btn", disabled=(gpt_client is None))
        ask_clicked = col2.button("질문 보내기", use_container_width=True,
                                key="ask_gpt_btn", disabled=(gpt_client is None))

        if gpt_client is None:
            st.info("API 키가 없어 비활성화되었습니다. 환경변수 OPENAI_API_KEY를 설정하거나 gpt_client 초기화를 확인하세요.")
        else:
            # 컨텍스트는 클릭 시점에 '있는 값만' 안전하게 구성
            def _safe_ctx():
                return build_gpt_context(
                    ahu=st.session_state.get("선택공조기_탭", "미선택"),
                    start_dt=locals().get("시작", pd.Timestamp.now()) if "시작" in locals() else pd.Timestamp.now(),
                    end_dt=locals().get("종료", pd.Timestamp.now()),
                    daily_df=locals().get("daily", pd.DataFrame()),
                    baseline_df=locals().get("ref", pd.DataFrame()),
                    k_sigma=locals().get("k_sigma", 1.0),
                    highlight_year=locals().get("highlight_year", None),
                    topN_df=locals().get("topN", pd.DataFrame()),
                )

            if gen_clicked:
                with st.spinner("GPT가 분석 중입니다..."):
                    ctx_text = _safe_ctx()
                    msgs = [
                        {"role": "system", "content":
                            "너는 공조기 에너지/비용 데이터 분석가야. "
                            "요약 데이터를 바탕으로 이상 원인, 절감 포인트, 계절 트렌드를 보수적으로 bullet로 정리해."},
                        {"role": "user", "content": f"컨텍스트:\n{ctx_text}\n\n요청: 핵심 인사이트 5개 + 권장 후속 분석 2개."}
                    ]
                    try:
                        resp = gpt_client.chat.completions.create(
                            model=GPT_MODEL, messages=msgs, temperature=0.3, max_tokens=700
                        )
                        st.markdown(resp.choices[0].message.content)
                    except Exception as e:
                        st.error(f"GPT 호출 오류: {e}")

            if ask_clicked:
                if not user_q.strip():
                    st.warning("질문을 입력하세요.")
                else:
                    with st.spinner("GPT가 답변 중..."):
                        ctx_text = _safe_ctx()
                        msgs = [
                            {"role": "system", "content": "너는 공조기 에너지/비용 데이터 분석가야. 간결하고 실행 가능한 답변을 해."},
                            {"role": "user", "content": f"배경 요약:\n{ctx_text}\n\n질문:\n{user_q}"}
                        ]
                        try:
                            resp = gpt_client.chat.completions.create(
                                model=GPT_MODEL, messages=msgs, temperature=0.3, max_tokens=600
                            )
                            st.markdown(resp.choices[0].message.content)
                        except Exception as e:
                            st.error(f"GPT 호출 오류: {e}")
    # ─────────────────────────────────────────────────────────



    # 2) 공조기 선택 (탭 안쪽, 날짜 밑)
    # [수정됨] Empty DataFrame 체크 추가
    ahu_set = set()
    if not df_final_all.empty and "공조기" in df_final_all.columns:
        ahu_set.update(df_final_all["공조기"].dropna().unique())
    if not all_df.empty and "공조기" in all_df.columns:
        ahu_set.update(all_df["공조기"].dropna().unique())

    공조기목록 = sorted(ahu_set) if ahu_set else ["AHU01"]
    선택공조기 = st.selectbox("📌 분석할 공조기 선택", 공조기목록, index=0, key="선택공조기_탭")

    # 3) 선택된 공조기 데이터 필터링
    # [수정됨] Empty DataFrame 체크 추가
    if not all_df.empty and "공조기" in all_df.columns and "datetime" in all_df.columns:
        df_ahu = all_df[
            (all_df["공조기"] == 선택공조기) &
            (all_df["datetime"] >= 시작) &
            (all_df["datetime"] < 종료)
        ].copy()
    else:
        df_ahu = pd.DataFrame()

    if df_ahu.empty:
        st.error("해당 공조기의 데이터가 없습니다.")
        st.stop()

    # 4) 공조기 형식 표시
    ahu_형식 = "일반형"
    if 선택공조기 in 건식제습형_공조기:
        ahu_형식 = "건식제습형"
    elif 선택공조기 in 냉각제습형_공조기:
        ahu_형식 = "냉각제습형"
    st.caption(f"📘 현재 선택된 공조기 형식: {ahu_형식}")

    # 5) 주요 항목 반복 시각화
    raw = load_ahu_detail_by_mode(선택공조기, mode)
    for 선택항목 in ["CCV", "PC_CCV", "HCV", "DH_HCV", "RAT", "RAH"]:
        if not raw.empty:
            df_selected = raw[raw["항목명"] == 선택항목].copy()
            if df_selected.empty:
                continue
            # 👉 여기서 그래프 함수 호출

    st.subheader("💸 에너지 비용 및 일일 비용 트렌드")

    ahu = 선택공조기

    # ▷ 기준 상향 옵션 (하이라이트 선택 제거)
    k_sigma = st.slider("기준 상향: 평균 + K·σ (K)", min_value=0.0, max_value=3.0, value=1.5, step=0.1)
    st.metric("기준 (월-일 평균 + K·σ)", f"K = {k_sigma:.1f}")

    # 1) 선택 기간 + 공조기 필터
    df_sel = all_df[
        (all_df["공조기"] == ahu) &
        (all_df["datetime"] >= 시작) &
        (all_df["datetime"] <  종료)
    ].copy()

    if df_sel.empty:
        st.info("해당 기간에 데이터가 없습니다.")
    else:
        # 2) 일일 총비용(원)
        cost_cols = [c for c in ["전력_비용(원)","비용(원)_CCV","비용(원)_PC_CCV","비용(원)_HCV","비용(원)_DH_HCV"] if c in df_sel.columns]
        if not cost_cols:
            st.info("비용 컬럼이 없어 절감분 분석을 건너뜁니다.")
        else:
            df_sel[cost_cols] = df_sel[cost_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            df_sel["날짜"] = df_sel["datetime"].dt.normalize()
            daily = df_sel.groupby("날짜", as_index=False)[cost_cols].sum()
            daily["총비용(원)"] = daily[cost_cols].sum(axis=1)

            # (선택) kWh
            kwh_cols = [c for c in ["kWh_CCV","kWh_PC_CCV","kWh_HCV","kWh_DH_HCV"] if c in df_sel.columns]
            if kwh_cols:
                df_sel[kwh_cols] = df_sel[kwh_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
                daily_kwh = df_sel.groupby("날짜", as_index=False)[kwh_cols].sum()
                daily_kwh["총kWh"] = daily_kwh[kwh_cols].sum(axis=1)
                daily = daily.merge(daily_kwh[["날짜","총kWh"]], on="날짜", how="left")
            else:
                daily["총kWh"] = np.nan

            # 3) 키/정렬용 월-일 문자열 (x축은 항상 01-01 ~ 12-31)
            daily["연도"] = pd.to_datetime(daily["날짜"]).dt.year
            daily["월일"] = pd.to_datetime(daily["날짜"]).dt.strftime("%m-%d")

            # 모든 월일 순서 고정 (카테고리 축용)
            monthday_order = pd.date_range("2000-01-01", "2000-12-31", freq="D").strftime("%m-%d").tolist()
            # 월별 눈금(1일만)
            month_ticks = pd.date_range("2000-01-01", "2000-12-01", freq="MS").strftime("%m-%d").tolist()

            # 4) 기준선: 월-일 평균 + Kσ
            ref = daily.groupby("월일", as_index=False).agg(
                평균비용=("총비용(원)", "mean"),
                표준편차=("총비용(원)", "std"),
                샘플수=("총비용(원)", "count")
            )
            ref["기준선(원)"] = ref["평균비용"] + k_sigma * ref["표준편차"].fillna(0)

            # 병합 & 초과분
            daily = daily.merge(ref[["월일","기준선(원)"]], on="월일", how="left")
            daily["초과비용(원)"] = (daily["총비용(원)"] - daily["기준선(원)"]).clip(lower=0)

            # 요약 표시
            st.metric("잠재 절감비용 (합계)", f"{int(round(float(daily['초과비용(원)'].sum()))):,} 원")
            if daily["총kWh"].notna().any():
                ref_e = daily.groupby("월일", as_index=False).agg(평균kWh=("총kWh","mean"), 표준편차kWh=("총kWh","std"))
                ref_e["기준kWh"] = ref_e["평균kWh"] + k_sigma * ref_e["표준편차kWh"].fillna(0)
                daily = daily.merge(ref_e[["월일","기준kWh"]], on="월일", how="left")
                daily["초과kWh"] = (daily["총kWh"] - daily["기준kWh"]).clip(lower=0)
                st.metric("잠재 절감에너지 (합계)", f"{int(round(float(daily['초과kWh'].sum()))):,} kWh")
            else:
                st.metric("잠재 절감에너지 (합계)", "—")

            # 5) 그래프: 연도 라인 + 기준선 + 각 연도 초과 구간 자동 음영
            # 라인
            # ── 기준치/라인/음영 그래프 (월일 기준, 1~12월 x축 고정 정렬) ─────────────────
            # daily: [날짜, 월일("MM-DD"), 연도, 총비용(원), 기준선(원)] 가 있어야 함

            # 1) 월일 카테고리 정렬 배열 (윤년 포함 366일 대비)
            월일_정렬 = pd.date_range("2001-01-01", "2001-12-31", freq="D").strftime("%m-%d").tolist()

            # ── 7) 연도 겹쳐보기 + 연도별 초과구간 음영(기준선 대비) ─────────────────
            pivot = daily.pivot_table(index="월일", columns="연도", values="총비용(원)", aggfunc="sum")
            pivot = pivot[[c for c in [2021, 2022, 2023, 2024, 2025] if c in pivot.columns]]
            if pivot.empty:
                st.info("그래프로 표시할 데이터가 없습니다.")
            else:
                # ① 모든 연도 라인 (겹쳐보기)
                plot_df = pivot.reset_index().melt(id_vars="월일", var_name="연도", value_name="총비용(원)")
                # x축 월-일 고정 순서
                monthday_order = sorted(plot_df["월일"].unique())
                fig = px.line(
                    plot_df.sort_values(["연도", "월일"]),
                    x="월일", y="총비용(원)", color=plot_df["연도"].astype(str),
                    title=f"{ahu} | 연도 겹쳐보기 (일일 총비용, 기준선={k_sigma:.1f}σ)",
                    markers=False
                )
                fig.update_layout(
                    xaxis=dict(
                        tickangle=-45,
                        categoryorder="array",
                        categoryarray=monthday_order
                    )
                )

                # ② 기준선(월-일 평균 + Kσ) 라인
                baseline_df = ref.sort_values("월일")
                fig.add_scatter(
                    x=baseline_df["월일"], y=baseline_df["기준선(원)"],
                    mode="lines", name="기준선(평균+K·σ)", line=dict(dash="dash"),
                    hoverinfo="skip"
                )

                # ── 보조: 색을 알파값이 있는 rgba로 바꾸는 유틸 ──
                def _to_rgba(color: str, alpha: float = 0.25) -> str:
                    if not isinstance(color, str):
                        return color
                    c = color.strip()
                    if c.startswith("rgba("):
                        # rgba(…, a) → a만 교체
                        body = c[5:-1]
                        parts = [p.strip() for p in body.split(",")]
                        if len(parts) == 4:
                            parts[-1] = str(alpha)
                            return "rgba(" + ", ".join(parts) + ")"
                        return c
                    if c.startswith("rgb("):
                        # rgb(r,g,b) → rgba(r,g,b,alpha)
                        return c.replace("rgb(", "rgba(").replace(")", f", {alpha})")
                    # 그 외(#색상 등)는 그대로 사용
                    return c

                # ③ 연도별 초과구간 음영 — 불연속 구간(run) 단위로 영역 채우기
                def add_exceed_fill_for_year(fig, df_year, year, color):
                    if not {"월일", "총비용(원)"}.issubset(df_year.columns):
                        return

                    d = df_year.copy()
                    d = d.sort_values("월일")

                    # 기준선 컬럼이 없으면 이 연도 자체의 월-일 평균으로 생성
                    if "기준선(원)" not in d.columns:
                        base = d.groupby("월일")["총비용(원)"].mean()
                        d["기준선(원)"] = d["월일"].map(base)

                    # ⬇⬇ 여기부터 수정된 부분 ⬇⬇
                    # 초과 여부 (pyarrow → 일반 bool로 변환)
                    over = (d["총비용(원)"] > d["기준선(원)"]).astype("bool")
                    if not over.any():
                        return

                    # 구간이 바뀌는 지점: bool → int64로 바꿔서 cumsum
                    change = (over != over.shift()).astype("int64")
                    run = change.cumsum()
                    # ⬆⬆ 여기까지 수정된 부분 ⬆⬆

                    shown = False
                    for _, g in d[over].groupby(run):
                        if g.empty:
                            continue

                        # 기준선 라인 (투명, fill 기반)
                        fig.add_scatter(
                            x=g["월일"],
                            y=g["기준선(원)"],
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                            connectgaps=False,
                        )

                        # 초과 구간 채우기
                        fig.add_scatter(
                            x=g["월일"],
                            y=g["총비용(원)"],
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            connectgaps=False,
                            name=f"{year} 초과구간(절감가능)",
                            fillcolor=_to_rgba(color, 0.25),
                            showlegend=not shown,
                        )
                        shown = True


                # plotly가 자동 배정한 선색을 재사용
                color_map = {
                    trace.name: getattr(trace.line, "color", None)
                    for trace in fig.data
                    if getattr(trace, "mode", "") == "lines" and trace.name not in ["기준선(평균+K·σ)"]
                }

                # 연도별로 초과구간 채우기
                for y in sorted(daily["연도"].unique()):
                    d_y = daily[daily["연도"] == y].copy()
                    if d_y.empty:
                        continue

                    # 혹시라도 기준선 컬럼이 빠져 있으면 ref에서 다시 붙여줌
                    if "기준선(원)" not in d_y.columns and "기준선(원)" in ref.columns:
                        d_y = d_y.merge(ref[["월일", "기준선(원)"]], on="월일", how="left")

                    c = color_map.get(str(y), "rgb(200,200,200)")
                    add_exceed_fill_for_year(fig, d_y, y, c)

                st.plotly_chart(fig, use_container_width=True)


    # ========================================================================

    # ✅ final_analysis parquet 기반 데이터만 사용
    # [수정됨] Empty DataFrame 체크 추가
    if not df_final_all.empty and "공조기" in df_final_all.columns and "datetime" in df_final_all.columns:
        df_ahu_final = df_final_all[
            (df_final_all["공조기"] == 선택공조기)
            & (df_final_all["datetime"] >= 시작)
            & (df_final_all["datetime"] < 종료)
        ].copy()
    else:
        df_ahu_final = pd.DataFrame()

    if df_ahu_final.empty:
        st.warning("해당 기간에 데이터가 없습니다.")
        st.stop()


    st.subheader("🌤️ 외기 조건 기반 유사일(Nearest Day) 비교")

    # 0) 외기 일평균 (일자 단위)
    if 외기df_daily is None or 외기df_daily.empty:
        st.info("외기(일평균) 데이터가 없어 유사일 분석을 건너뜁니다.")
        st.stop()

    oa_daily = 외기df_daily.copy()
    oa_daily["날짜"] = oa_daily["datetime"].dt.normalize()
    oa_daily["연도"] = oa_daily["날짜"].dt.year

    # 1) AHU 일일 총비용/총kWh 계산 (선택 기간 + 선택 공조기)
    # [수정됨] Empty DataFrame 체크 추가
    if not all_df.empty and "공조기" in all_df.columns and "datetime" in all_df.columns:
        df_sel = all_df[
            (all_df["공조기"] == 선택공조기) &
            (all_df["datetime"] >= 시작) &
            (all_df["datetime"] <  종료)
        ].copy()
    else:
        df_sel = pd.DataFrame()

    if df_sel.empty:
        st.info("선택한 공조기의 해당 기간 데이터가 없습니다.")
        st.stop()

    cost_cols = [c for c in ["전력_비용(원)","비용(원)_CCV","비용(원)_PC_CCV","비용(원)_HCV","비용(원)_DH_HCV"] if c in df_sel.columns]
    if not cost_cols:
        st.info("비용 컬럼이 없어 유사일 분석을 건너뜁니다.")
        st.stop()

    df_sel[cost_cols] = df_sel[cost_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df_sel["날짜"] = df_sel["datetime"].dt.normalize()

    daily_cost = df_sel.groupby("날짜", as_index=False)[cost_cols].sum()
    daily_cost["총비용(원)"] = daily_cost[cost_cols].sum(axis=1)

    kwh_cols = [c for c in ["kWh_CCV","kWh_PC_CCV","kWh_HCV","kWh_DH_HCV"] if c in df_sel.columns]
    if kwh_cols:
        df_sel[kwh_cols] = df_sel[kwh_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        daily_kwh = df_sel.groupby("날짜", as_index=False)[kwh_cols].sum()
        daily_kwh["총kWh"] = daily_kwh[kwh_cols].sum(axis=1)
        daily = daily_cost.merge(daily_kwh[["날짜","총kWh"]], on="날짜", how="left")
    else:
        daily = daily_cost.assign(총kWh=np.nan)

    daily["연도"] = daily["날짜"].dt.year

    # 2) 타겟 외기 조건 입력 (오늘값 자동 or 수동)
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        모드 = st.radio("타겟 외기 조건", ["오늘값 사용", "직접 입력"], horizontal=True)
    if 모드 == "오늘값 사용":
        # oa_daily 가장 최신 날짜의 평균을 사용
        last_row = oa_daily.sort_values("날짜").tail(1)
        target_T = float(last_row["외기온도"].iloc[0]) if not last_row.empty else 23.0
        target_H = float(last_row["외기습도"].iloc[0]) if not last_row.empty else 57.0
    else:
        with c2:
            target_T = st.number_input("타겟 외기온도(℃)", value=23.0, step=0.1, format="%.1f")
        with c3:
            target_H = st.number_input("타겟 외기습도(%)", value=57.0, step=0.1, format="%.1f")

    c4, c5, c6 = st.columns([1,1,2])
    with c4:
        metric = st.selectbox("비교 지표", ["총비용(원)","총kWh"])
    with c5:
        # [수정됨] Empty DataFrame 체크 추가
        if not all_df.empty and "datetime" in all_df.columns:
            year_min = int(all_df["datetime"].dt.year.min())
            year_max = int(all_df["datetime"].dt.year.max())
            years = list(range(year_min, year_max+1))
        else:
            years = [2025]
        선택연도 = st.multiselect("비교 연도", years, default=[y for y in range(2021, 2026) if y in years])
    with c6:
        방법 = st.radio("선정 방식", ["가까운 1일(거리 최소)","허용 오차 내 평균"], horizontal=True)

    # 허용 오차 설정(선택)
    tol_col1, tol_col2 = st.columns(2)
    with tol_col1:
        tol_T = st.number_input("허용 오차(온도, ℃)", value=1.0, step=0.1, format="%.1f")
    with tol_col2:
        tol_H = st.number_input("허용 오차(습도, %)", value=5.0, step=0.1, format="%.1f")

    # 3) 유사일 매칭 함수
    def pick_similar_by_year(oa_daily_df, target_T, target_H, years, method="nearest", tol_T=1.0, tol_H=5.0):
        """
        method:
          - 'nearest' : 연도별로 거리(온도/습도 차이 제곱합) 가장 작은 1일 선택
          - 'tolerance_mean' : tol 범위(온도/습도) 안의 날짜들 평균 사용
        """
        results = []
        for y in years:
            cand = oa_daily_df[oa_daily_df["연도"] == y].copy()
            if cand.empty: 
                continue

            cand["dT"] = cand["외기온도"] - target_T
            cand["dH"] = cand["외기습도"] - target_H
            # 간단한 가중 유클리드 거리(정규화 없이)
            cand["dist"] = np.sqrt(cand["dT"]**2 + (cand["dH"]/2.0)**2)  # 습도 영향 조금 낮춤

            if method == "nearest":
                row = cand.loc[cand["dist"].idxmin()]
                results.append({
                    "연도": y,
                    "날짜": row["날짜"],
                    "외기온도": row["외기온도"],
                    "외기습도": row["외기습도"],
                    "dist": row["dist"],
                    "선정방식": "가까운 1일"
                })
            else:
                subset = cand[(cand["dT"].abs() <= tol_T) & (cand["dH"].abs() <= tol_H)]
                if subset.empty:
                    # 오차내가 없으면 최근접 1일로 대체
                    row = cand.loc[cand["dist"].idxmin()]
                    results.append({
                        "연도": y,
                        "날짜": row["날짜"],
                        "외기온도": row["외기온도"],
                        "외기습도": row["외기습도"],
                        "dist": row["dist"],
                        "선정방식": "최근접(대체)"
                    })
                else:
                    row = subset.sort_values("dist").head(1).iloc[0]
                    # 평균치도 같이 제공(선택일 포함)
                    results.append({
                        "연도": y,
                        "날짜": row["날짜"],
                        "외기온도": subset["외기온도"].mean(),
                        "외기습도": subset["외기습도"].mean(),
                        "dist": subset["dist"].mean(),
                        "선정방식": f"오차내 평균({len(subset)}일)"
                    })
        return pd.DataFrame(results)

    method_key = "nearest" if 방법 == "가까운 1일(거리 최소)" else "tolerance_mean"
    picked = pick_similar_by_year(oa_daily, target_T, target_H, 선택연도, method=method_key, tol_T=tol_T, tol_H=tol_H)

    if picked.empty:
        st.info("선택된 연도에서 유사한 외기조건의 날짜를 찾지 못했습니다.")
        st.stop()

    # 4) 유사일과 AHU 일일 지표(총비용/총kWh) 결합
    merged = picked.merge(daily[["날짜","연도",metric]], on=["날짜","연도"], how="left")
    merged = merged.dropna(subset=[metric])  # 지표 없는 날 제거
    merged = merged.sort_values("연도")

    # 5) 시각화
    title = f"{선택공조기} | 타겟 외기 {target_T:.1f}℃ / {target_H:.1f}% 유사일 비교 ({metric})"
    if merged.empty:
        st.info("유사일의 AHU 지표가 없어 표시할 데이터가 없습니다.")
    else:
        fig = px.line(
            merged, x="연도", y=metric, markers=True,
            title=title, text="연도"
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        # 보조: 실제 매칭된 날짜/외기/거리 표
        show = merged.copy()
        show["날짜"] = show["날짜"].dt.strftime("%Y-%m-%d")
        if metric.endswith("(원)"):
            show[metric] = show[metric].apply(lambda x: f"{int(round(x)):,}")
        else:
            show[metric] = show[metric].apply(lambda x: f"{x:,.1f}")
        show["dist"] = show["dist"].apply(lambda x: f"{x:.2f}")
        st.markdown("#### 매칭 결과표")
        st.dataframe(
            show[["연도","날짜","외기온도","외기습도","dist","선정방식",metric]],
            use_container_width=True
        )

        st.caption("※ 동일 외기조건이 없으면 거리(온도·습도 차)로 가장 가까운 날짜를 선택합니다. ‘허용 오차 내 평균’을 선택하면 범위 내 여러 날짜의 평균값으로 비교합니다.")

    # 1) 항목별 총 비용 산출 (CCV, PC_CCV, HCV, DH_HCV)
    항목리스트 = ["CCV", "PC_CCV", "HCV", "DH_HCV"]

    # 2) 일자 단위 비용 트렌드 (단순 선 그래프)
    cols = [c for c in df_ahu_final.columns if any(c.endswith(f"_{h}") for h in 항목리스트)]

    if cols:
        daily_cost = (
            df_ahu_final.groupby(df_ahu_final["datetime"].dt.date)[cols].sum().reset_index()
            .melt(id_vars="datetime", var_name="항목", value_name="값")
        )
        daily_cost["항목명"] = daily_cost["항목"].str.split("_").str[-1]
        daily_cost["지표"] = daily_cost["항목"].str.split("_").str[0]  # kWh or 비용(원)
        daily_cost = daily_cost.pivot_table(
            index=["datetime","항목명"], columns="지표", values="값", aggfunc="sum"
        ).reset_index().rename(columns={"datetime":"날짜"})
    else:
        # 컬럼이 하나도 없으면 빈 DataFrame 리턴
        daily_cost = pd.DataFrame(columns=["날짜","항목명","kWh","비용(원)"])

    if daily_cost.empty:
        st.info("📅 선택 기간에 일별 비용 데이터가 없습니다.")
    else:
        # 이전에 발생했던 오류 수정
        if '날짜' not in daily_cost.columns:
            daily_cost = daily_cost.rename(columns={daily_cost.columns[0]: '날짜'})
            
        fig = px.line(
            daily_cost,
            x="날짜", y="비용(원)", color="항목명",
            title=f"{ahu} 일별 비용 추이",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3) pivot 기반 일자별 비용 집계 (총비용 포함)
    cols = [f"비용(원)_{h}" for h in ["CCV","PC_CCV","HCV","DH_HCV"] if f"비용(원)_{h}" in df_ahu_final.columns]

    if cols:
        daily_total = (
            df_ahu_final.groupby(df_ahu_final["datetime"].dt.date)[cols].sum().reset_index()
            .melt(id_vars="datetime", var_name="항목", value_name="비용(원)")
        )
        daily_total["항목명"] = daily_total["항목"].str.replace("비용\\(원\\)_","",regex=True)
        daily_total.rename(columns={"datetime":"날짜"}, inplace=True)
    else:
        daily_total = pd.DataFrame(columns=["날짜","항목","항목명","비용(원)"])

    항목별_일일비용_그래프 = []
    if not daily_total.empty:
        pivot_daily = daily_total.pivot_table(
            index="날짜",  # 공조기별이면 위에서 컬럼 추가 필요
            columns="항목명",
            values="비용(원)",
            aggfunc="sum",
            fill_value=0
        ).reset_index()

        # ✅ 총비용 계산
        pivot_daily["총비용(원)"] = (
            pivot_daily.get("CCV",0)
            + pivot_daily.get("PC_CCV",0)
            + pivot_daily.get("HCV",0)
            + pivot_daily.get("DH_HCV",0)
        )

        # ✅ tidy 형태 변환
        for 항목 in ["CCV","PC_CCV","HCV","DH_HCV","총비용(원)"]:
            if 항목 in pivot_daily.columns:
                df_plot = pivot_daily[["날짜", 항목]].rename(columns={항목:"비용(원)"})
                항목별_일일비용_그래프.append((ahu, 항목, df_plot))

        # 4) 공조기별 비용 요약표 (선택 공조기 1대 기준)
        # all_df_range(기간 필터 적용 전체)에서 동일 로직 재사용
        summary_raw = make_top_summary(all_df_range, raw=True)

        df_cost = summary_raw[summary_raw["공조기"] == 선택공조기].copy()
        if df_cost.empty:
            st.subheader("📊 공조기별 비용 요약표")
            st.info("선택한 공조기에 대한 집계 결과가 없습니다.")
        else:
            st.subheader("📊 공조기별 비용 요약표")
            # [수정됨] Debug print 제거 (empty DataFrame 로그 방지)
            df_cost_fmt = df_cost.copy()
            df_cost_fmt.index = np.arange(1, len(df_cost_fmt) + 1)
            df_cost_fmt.index.name = "No"

            money_cols = [c for c in df_cost_fmt.columns if c.endswith("(원)")]
            # [수정됨] applymap deprecation 대응 (Series.map 사용)
            df_cost_fmt[money_cols] = df_cost_fmt[money_cols].apply(
                lambda s: s.map(lambda x: f"{int(round(x)):,}" if pd.notna(x) else "")
            )

            컬럼_색상맵 = {
                "총비용(원)": "#e6ffe6",
                "전력사용량(원)": "#fff5e6",
                "냉수코일비용(원)": "#e6eeff",
                "스팀코일비용(원)": "#ffe6e6",
            }

            def style_col_background(col):
                color = 컬럼_색상맵.get(col.name, "")
                return [f"background-color: {color}"] * len(col)

            styled = (
                df_cost_fmt.style
                .apply(style_col_background, subset=[c for c in df_cost_fmt.columns if c in 컬럼_색상맵])
            )
            st.dataframe(styled, use_container_width=True)


    # 5) 장치별 전기사용량
    for ahu in [선택공조기]:
        # [수정됨] Empty DataFrame 체크 추가
        if not all_df.empty and "공조기" in all_df.columns and "datetime" in all_df.columns:
            # all_df를 기반으로 필터링
            df_filt = all_df[
                (all_df["공조기"] == ahu)
                & (all_df["datetime"] >= 시작)
                & (all_df["datetime"] < 종료)
            ].copy()
        else:
            df_filt = pd.DataFrame()

        if df_filt.empty:
            continue
            
        rows = []
        전력_cols = [
            c for c in df_filt.columns
            if c.startswith("kWh_")
            and c.split("_", 1)[1] not in ("냉수", "스팀")   # kWh_냉수 / kWh_스팀 제외
        ]

        rows = []
        for col in 전력_cols:
            장치 = col.split("_", 1)[1]  # 예: "kWh_SFST1" -> "SFST1"
            kwh = df_filt[col].sum()

            h_col = f"운전시간(h)_{장치}"     # 피벗 결과는 '운전시간(h)_SFST1' 형태
            hours = df_filt[h_col].sum() if h_col in df_filt.columns else 0.0

            cost_col = f"비용(원)_{장치}"      # '비용(원)_SFST1'
            cost = df_filt[cost_col].sum() if cost_col in df_filt.columns else None

            if kwh > 0:
                rows.append({
                    "장치": 장치,
                    "가동시간(h)": round(hours, 1),
                    "사용량(kWh)": int(round(kwh)),
                    "비용(원)": int(round(cost)) if cost is not None else None
                })

        if not rows:
            continue

        df_장치 = pd.DataFrame(rows)
        장치_이름_한글 = {
            "SF": "서플라이팬", "AC_SFST": "서플라이팬",
            "PC_SFSS": "프로세스팬", "OAU_SFST": "프로세스팬", "RFST": "프로세스팬", "PC_SFST": "프로세스팬",
            "EFSS": "배기팬", "EFST": "배기팬", "AC_RFSS": "배기팬",
            "CDU": "CDU", "CDUSS": "CDU", "COMP": "CDU",
            "EH": "히터", "HT": "히터", "EHSS1": "히터1", "EHSS2": "히터2", "EHSS3": "히터3"
        }
        df_장치["장치"] = df_장치["장치"].map(장치_이름_한글).fillna(df_장치["장치"])
        df_장치["가동시간(h)"] = df_장치["가동시간(h)"].map(lambda x: f"{x:.1f}")
        df_장치["사용량(kWh)"] = df_장치["사용량(kWh)"].map(lambda x: f"{x:,}")
        df_장치["비용(원)"] = df_장치["비용(원)"].map(lambda x: f"{x:,}")

        df_장치.index = df_장치.index + 1
        df_장치.index.name = "No"

        st.markdown(f"#### {ahu} 장치별 전기사용량 및 전기비용")
        st.dataframe(df_장치, use_container_width=True)
        st.markdown("---")

    # 6) 항목별/총비용 그래프 출력
    for ahu, 항목명, 일별 in 항목별_일일비용_그래프:
        일별 = 일별.copy()

         # ✅ 공조기 컬럼 보강
        일별["공조기"] = ahu

        # ✅ 날짜 컬럼 안전 처리
        if "datetime" in 일별.columns:
            일별["날짜"] = pd.to_datetime(일별["datetime"])
        elif "날짜" in 일별.columns:
            일별["날짜"] = pd.to_datetime(일별["날짜"])
        else:
            raise KeyError("일별 데이터에 날짜 컬럼이 없습니다.")

        일별["비용(만원)"] = 일별["비용(원)"] / 10000
        일별["연도"] = 일별["날짜"].dt.year
        일별["절기"] = 일별["날짜"].apply(절기_분류)

        label = "총비용" if "총비용" in 항목명 else 항목명_한글.get(항목명, 항목명)
        expander_title = f"{label} - {ahu} | 절기별 연도별 일일 비용"

        with st.expander(f"📈 {expander_title}", expanded=("총비용" in label)):
            draw_season_year_line(
                일별,
                y_col="비용(만원)",
                title=expander_title,
                평균선_컬럼="비용(만원)"
            )
with 탭3:
    st.subheader("🔋 공조기별 에너지 사용량 상세 분석")

    st.caption("※ 상단에서 설정한 날짜 범위(📅 분석할 날짜 범위 선택)에 맞춰 공조기별 에너지 사용량을 집계합니다.")

    # 기간 필터 적용된 전체 DF 사용
    df_energy_base = all_df_range.copy()

    if df_energy_base.empty:
        st.warning("선택한 기간에 해당하는 데이터가 없습니다.")
        st.stop()

    # 필요한 컬럼이 없더라도 에러 나지 않도록 방어적으로 0 컬럼 생성
    for col in ["kWh_HCV", "kWh_DH_HCV", "kWh_CCV", "kWh_PC_CCV"]:
        if col not in df_energy_base.columns:
            df_energy_base[col] = 0.0

    # 전기 사용량(kWh) 컬럼 찾기: kWh_접두사 / _kWh 접미사 중에서 냉수/스팀/코일 계열 제외
    power_kwh_cols = [c for c in df_energy_base.columns if _is_power_kwh(c)]

    if power_kwh_cols:
        df_energy_base[power_kwh_cols] = df_energy_base[power_kwh_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0)
    else:
        # 장치별 kWh 컬럼이 하나도 없다면 전기 사용량은 0으로 처리
        df_energy_base["전기_가상kWh"] = 0.0
        power_kwh_cols = ["전기_가상kWh"]

    # 코일 kWh 숫자화
    kwh_cols_all = ["kWh_HCV", "kWh_DH_HCV", "kWh_CCV", "kWh_PC_CCV"]
    df_energy_base[kwh_cols_all] = df_energy_base[kwh_cols_all].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    # ─────────────────────────────────────────────
    # 1) 공조기별 에너지 사용량 집계 (kWh)
    # ─────────────────────────────────────────────
    grp = df_energy_base.groupby("공조기", as_index=False).agg(
        스팀_kWh=("kWh_HCV", "sum"),
        제습용_스팀_kWh=("kWh_DH_HCV", "sum"),
        냉수_kWh=("kWh_CCV", "sum"),
        프리쿨러_냉수_kWh=("kWh_PC_CCV", "sum"),
    )

    # 전기 사용량 kWh = power_kwh_cols 합
    df_power = (
        df_energy_base[["공조기"] + power_kwh_cols]
        .groupby("공조기", as_index=False)[power_kwh_cols]
        .sum()
    )
    df_power["전기_kWh"] = df_power[power_kwh_cols].sum(axis=1)

    # 병합
    grp = grp.merge(df_power[["공조기", "전기_kWh"]], on="공조기", how="left")

    # 총 에너지(비교용)
    grp["총_kWh"] = (
        grp["스팀_kWh"]
        + grp["제습용_스팀_kWh"]
        + grp["냉수_kWh"]
        + grp["프리쿨러_냉수_kWh"]
        + grp["전기_kWh"]
    )

    # [수정됨] 비율(%) 계산은 숫자 컬럼 상태에서 먼저 수행
    df_display = grp.copy()
    for col in ["스팀_kWh", "제습용_스팀_kWh", "냉수_kWh", "프리쿨러_냉수_kWh", "전기_kWh"]:
        ratio_col = col.replace("_kWh", "_비중(%)")
        df_display[ratio_col] = np.where(
            df_display["총_kWh"] > 0,
            df_display[col] / df_display["총_kWh"] * 100,
            0,
        ).round(1)

    # 표시용 포맷 (천 단위 콤마 & 원본 보존)
    for col in ["스팀_kWh", "제습용_스팀_kWh", "냉수_kWh", "프리쿨러_냉수_kWh", "전기_kWh", "총_kWh"]:
        df_display[col + "_raw"] = df_display[col]  # 원본 값 보존
        df_display[col] = df_display[col].apply(lambda x: f"{x:,.1f}")

    # 표시용 컬럼만 선택
    cols_order = [
        "공조기",
        "스팀_kWh", "스팀_비중(%)",
        "제습용_스팀_kWh", "제습용_스팀_비중(%)",
        "냉수_kWh", "냉수_비중(%)",
        "프리쿨러_냉수_kWh", "프리쿨러_냉수_비중(%)",
        "전기_kWh", "전기_비중(%)",
        "총_kWh",
    ]
    # 실제 존재하는 컬럼만 사용
    cols_order = [c for c in cols_order if c in df_display.columns]

    st.markdown("### 📘 공조기별 에너지 사용량 요약 (kWh 기준)")
    df_display_show = df_display[cols_order].copy()
    df_display_show.index = range(1, len(df_display_show) + 1)
    df_display_show.index.name = "No"
    st.dataframe(df_display_show, use_container_width=True)

    st.caption("※ kWh 기준으로 스팀/제습 스팀/냉수/프리쿨러 냉수/전기 사용량과 각 비중(%)을 표시합니다.")

    # ─────────────────────────────────────────────
    # 2) 공조기별 에너지 사용량 비교 (막대 그래프)
    # ─────────────────────────────────────────────
    # tidy 형태로 melt
    melt_cols = ["스팀_kWh", "제습용_스팀_kWh", "냉수_kWh", "프리쿨러_냉수_kWh", "전기_kWh"]
    melt_cols = [c for c in melt_cols if c in grp.columns]

    df_bar = grp[["공조기"] + melt_cols].copy()
    df_bar_melt = df_bar.melt(id_vars="공조기", var_name="에너지종류", value_name="kWh")
    df_bar_melt["kWh"] = df_bar_melt["kWh"].fillna(0)

    st.markdown("### 📊 공조기별 에너지 사용량 비교")

    fig_bar = px.bar(
        df_bar_melt,
        x="공조기",
        y="kWh",
        color="에너지종류",
        barmode="stack",
        title="공조기별 에너지 사용량(스택형, kWh)",
    )
    fig_bar.update_layout(xaxis_title="공조기", yaxis_title="에너지 사용량 (kWh)")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ─────────────────────────────────────────────
    # 3) 특정 공조기 선택 후 일별 에너지 사용량 추이
    # ─────────────────────────────────────────────
    st.markdown("### 📈 선택 공조기 일별 에너지 사용량 추이")

    선택공조기_에너지 = st.selectbox(
        "상세 분석할 공조기를 선택하세요",
        sorted(grp["공조기"].unique()),
        key="선택공조기_에너지_탭3",
    )

    df_ahu_energy = df_energy_base[df_energy_base["공조기"] == 선택공조기_에너지].copy()
    if df_ahu_energy.empty:
        st.info("선택한 공조기의 해당 기간 데이터가 없습니다.")
    else:
        df_ahu_energy["날짜"] = df_ahu_energy["datetime"].dt.normalize()

        # 일별 합산
        daily_ahu = df_ahu_energy.groupby("날짜", as_index=False).agg(
            스팀_kWh=("kWh_HCV", "sum"),
            제습용_스팀_kWh=("kWh_DH_HCV", "sum"),
            냉수_kWh=("kWh_CCV", "sum"),
            프리쿨러_냉수_kWh=("kWh_PC_CCV", "sum"),
        )

        df_power_daily = (
            df_ahu_energy[["날짜"] + power_kwh_cols]
            .groupby("날짜", as_index=False)[power_kwh_cols]
            .sum()
        )
        df_power_daily["전기_kWh"] = df_power_daily[power_kwh_cols].sum(axis=1)

        daily_ahu = daily_ahu.merge(df_power_daily[["날짜", "전기_kWh"]], on="날짜", how="left")
        daily_ahu = daily_ahu.fillna(0)

        # tidy 형태로 melt
        melt_cols_daily = ["스팀_kWh", "제습용_스팀_kWh", "냉수_kWh", "프리쿨러_냉수_kWh", "전기_kWh"]
        df_daily_melt = daily_ahu.melt(id_vars="날짜", var_name="에너지종류", value_name="kWh")
        df_daily_melt["kWh"] = df_daily_melt["kWh"].fillna(0)

        fig_line = px.line(
            df_daily_melt,
            x="날짜",
            y="kWh",
            color="에너지종류",
            markers=True,
            title=f"{선택공조기_에너지} 일별 에너지 사용량 추이 (kWh)",
        )
        fig_line.update_layout(xaxis_title="날짜", yaxis_title="에너지 사용량 (kWh)")
        st.plotly_chart(fig_line, use_container_width=True)

        with st.expander("📄 선택 공조기 일별 에너지 사용량 표 보기", expanded=False):
            df_daily_show = daily_ahu.copy()
            for col in ["스팀_kWh", "제습용_스팀_kWh", "냉수_kWh", "프리쿨러_냉수_kWh", "전기_kWh"]:
                if col in df_daily_show.columns:
                    df_daily_show[col] = df_daily_show[col].apply(lambda x: f"{x:,.1f}")
            st.dataframe(df_daily_show, use_container_width=True)






with 탭5:
    st.subheader("📊 항목별 요약 통계")

    # === 색상 & 항목명 매핑 ===
    항목_색상맵 = {
        "프리쿨러 냉수코일": "#f0faff",
        "냉수코일": "#e6eeff",
        "제습 스팀코일": "#f5d9c6",
        "스팀코일": "#ffe6e6",
        "환기온도": "#d3ebac",
        "환기습도": "#b9cfca"
    }
    항목명_정규화 = {
        "CCV": "냉수코일",
        "PC_CCV": "프리쿨러 냉수코일",
        "HCV": "스팀코일",
        "DH_HCV": "제습 스팀코일",
        "RAT": "환기온도",
        "RAH": "환기습도",
    }

    def style_by_항목(row, ref_df):
        color = 항목_색상맵.get(ref_df.loc[row.name, "항목_정규화"], "")
        return [f"background-color: {color}"] * len(row)

    def show_styled_dataframe(df_raw, name="표", show_index=True):
        ref_df = df_raw.copy()
        df = df_raw.drop(columns=["항목_정규화"])
        styled_df = df.style.apply(style_by_항목, axis=1, args=(ref_df,))
        st.markdown(name)
        st.dataframe(styled_df, use_container_width=True, hide_index=not show_index)

    # ✅ final_analysis parquet 기반 데이터
    items = get_items_from_final(all_df)
    target_items = ["CCV","PC_CCV","HCV","DH_HCV","RAT","RAH"]
    use_items = [i for i in target_items if i in items]

    # 최소 1개 이상 있을 때만 진행
    cols = []
    for h in use_items:
        for prefix in ["kWh","비용(원)","평균 개도율(%)"]:
            col = f"{prefix}_{h}"
            if col in all_df.columns:
                cols.append(col)

    if not cols:
        st.info("📊 선택된 기간에 해당 항목 데이터가 없습니다.")
        df_summary = pd.DataFrame(columns=["연도","절기","항목명","누적 열량(kWh)","평균값","평균 개도율(%)"])
    else:
        # long 형태로 변환 (딱 한 번만 melt)
        df_energy = all_df[["공조기","datetime"] + cols].copy()
        df_energy = df_energy.melt(id_vars=["공조기","datetime"], var_name="지표", value_name="값")
        df_energy["항목명"] = df_energy["지표"].str.split("_").str[-1]
        df_energy["지표타입"] = df_energy["지표"].str.split("_").str[0]  # 'kWh' / '비용(원)' / '평균 개도율(%)'
        df_energy["연도"] = df_energy["datetime"].dt.year
        df_energy["절기"] = df_energy["datetime"].apply(절기_분류)

        # 1) 코일 에너지(kWh) 누적 (여기서 'kWh' 컬럼이 아니라 값 컬럼을 합산)
        df_energy_kwh = (
            df_energy[
                (df_energy["항목명"].isin(["CCV","PC_CCV","HCV","DH_HCV"])) &
                (df_energy["지표타입"] == "kWh")
            ]
            .groupby(["연도","절기","항목명"], as_index=False)["값"]
            .sum()
            .rename(columns={"값": "누적 열량(kWh)"})
        )

        # 2) RAT/RAH 평균
        df_avg = (
            df_energy[df_energy["항목명"].isin(["RAT","RAH"])]
            .groupby(["연도","절기","항목명"], as_index=False)["값"]
            .mean()
            .rename(columns={"값": "평균값"})
        )

        # 3) 통합
        df_summary = pd.concat([df_energy_kwh, df_avg], ignore_index=True)

        # 항목명 한글화용 정규화 & 정렬
        df_summary["항목_정규화"] = df_summary["항목명"].map(항목명_정규화).fillna(df_summary["항목명"])
        df_summary = df_summary.sort_values(["연도","절기","항목_정규화"]).reset_index(drop=True)

        # 출력
        show_styled_dataframe(df_summary, name="📊 연도/절기별 요약 통계", show_index=False)

        # 🌬 환기온습도 추가 (detail parquet 기반)
        raw = load_ahu_detail_by_mode(선택공조기, mode)
        if raw is not None and not raw.empty:
            df_vent = raw[
                (raw["항목명"].isin(["RAT", "RAH"])) &
                (raw["datetime"] >= 시작) &
                (raw["datetime"] <  종료)
            ].copy()
            if not df_vent.empty:
                df_vent["연도"] = df_vent["datetime"].dt.year
                df_vent["절기"] = df_vent["datetime"].apply(절기_분류)
                df_vent_summary = (
                    df_vent.groupby(["연도", "절기", "항목명"])["값"]
                    .mean()
                    .reset_index()
                    .rename(columns={"값": "평균값"})
                )
                df_vent_summary["누적 열량(kWh)"] = None
                df_vent_summary["평균 개도율(%)"] = None
                df_summary = pd.concat([df_summary, df_vent_summary], ignore_index=True)


        # 항목명 한글화
        df_summary["항목명"] = df_summary["항목명"].map(항목명_정규화).fillna(df_summary["항목명"])


    # ---- 월별 평균 개도율(%) 계산 ----
    coil_items = ["CCV", "PC_CCV", "HCV", "DH_HCV"]

    raw = load_ahu_detail_by_mode(선택공조기, mode)

    if raw is not None and not raw.empty:
        # 코일 항목 + 기간 필터
        df_raw = raw[
            (raw["항목명"].isin(coil_items)) &
            (raw["datetime"] >= 시작) &
            (raw["datetime"] <  종료)
        ].copy()

        if df_raw.empty:
            월별_평균개도율 = pd.DataFrame(columns=["공조기","월","항목","평균 개도율(%)"])
        else:
            # 시간 가중치 계산 (샘플 간격 기반)
            df_raw = df_raw.sort_values(["항목명","datetime"])
            df_raw["dt_h"] = (
                df_raw.groupby("항목명")["datetime"]
                    .diff()
                    .dt.total_seconds()
                    .div(3600)
            )

            # 첫 샘플/비정상 간격 제거 (0 < dt <= 12h만 인정)
            df_raw = df_raw[(df_raw["dt_h"] > 0) & (df_raw["dt_h"] <= 12)].copy()
            df_raw["월"] = df_raw["datetime"].dt.to_period("M")

            # 시간가중 평균 개도율(%)
            def _wavg(g):
                return np.average(g["값"], weights=g["dt_h"])

            wavg = (
                df_raw.groupby(["월","항목명"])
                    .apply(_wavg)
                    .reset_index(name="평균 개도율(%)")
            )

            # 표 정리
            wavg["공조기"] = 선택공조기
            wavg["월"] = wavg["월"].astype(str)
            wavg["항목"] = wavg["항목명"].map(항목명_한글).fillna(wavg["항목명"])
            월별_평균개도율 = wavg[["공조기","월","항목","평균 개도율(%)"]]

    else:
        월별_평균개도율 = pd.DataFrame(columns=["공조기","월","항목","평균 개도율(%)"])

    # 👉 화면에 표시
    st.subheader("📌 월별 평균 개도율(%)")
    st.dataframe(월별_평균개도율, use_container_width=True)


    if not 월별_평균개도율.empty:
        연도별_평균개도율 = (
            월별_평균개도율
            .assign(연도=lambda x: x["월"].str[:4].astype(int))  # 🔹 여기서 바로 int로
            .groupby(["항목","연도"], as_index=False)["평균 개도율(%)"]
            .mean()
        )
    else:
        연도별_평균개도율 = pd.DataFrame(columns=["항목","연도","평균 개도율(%)"])

    # 연도별 실제 날짜 수
    일수_df = (
        df_energy.groupby(df_energy["datetime"].dt.year)
        .agg(일수=("datetime", "nunique"))
        .reset_index()
        .rename(columns={"datetime": "연도"})
    )

    # df_summary에는 '평균 개도율(%)' 아직 없음 → agg 대상에서 제외
    df_연도별 = (
        df_summary.groupby(["항목명", "연도"], as_index=False)
        .agg({
            "누적 열량(kWh)": "sum",
            "평균값": "mean"
        })
        .merge(일수_df, on="연도", how="left")
    )

    df_연도별 = df_연도별.rename(columns={"항목명": "항목"})

    # 평균 열량(kWh) 계산
    df_연도별["평균 열량(kWh)"] = df_연도별["누적 열량(kWh)"] / df_연도별["일수"]



    df_연도별 = df_연도별.rename(columns={"항목명": "항목", "날짜": "일수"})

    # 평균 열량 = 누적 열량 ÷ 실제 일수
    df_연도별["평균 열량(kWh)"] = df_연도별["누적 열량(kWh)"] / df_연도별["일수"]

    # 에너지 항목만 (코일)
    df_energy_only = df_연도별[~df_연도별["항목"].isin(["환기온도", "환기습도"])].copy()
    df_energy_only = df_energy_only[["항목","연도","누적 열량(kWh)","평균 열량(kWh)"]]  # ✅ 여기서 '평균 개도율(%)' 제거

    df_energy_only = df_energy_only.merge(연도별_평균개도율, on=["항목","연도"], how="left")
    # 숫자 포맷팅
    df_energy_only["누적 열량(kWh)"] = df_energy_only["누적 열량(kWh)"].apply(lambda x: f"{int(round(x)):,}")
    df_energy_only["평균 열량(kWh)"] = df_energy_only["평균 열량(kWh)"].apply(lambda x: f"{int(round(x)):,}")
    df_energy_only["평균 개도율(%)"] = df_energy_only["평균 개도율(%)"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    df_energy_only["항목_정규화"] = df_energy_only["항목"].map(항목명_정규화).fillna(df_energy_only["항목"])
    df_energy_only.index = range(1, len(df_energy_only) + 1)
    df_energy_only.index.name = "No"
    show_styled_dataframe(df_energy_only, "📌 **항목별 연도별 열량 및 평균 개도율**")

    # 🔹 환경 항목만 (RAT, RAH)
    df_env = df_연도별[df_연도별["항목"].isin(["환기온도", "환기습도"])].copy()
    df_env = df_env[["항목","연도","평균값"]]
    df_env["평균값"] = df_env.apply(
        lambda row: f"{row['평균값']:.2f} ℃" if row["항목"] == "환기온도" else f"{row['평균값']:.2f} %", axis=1
    )
    df_env["항목_정규화"] = df_env["항목"].map(항목명_정규화).fillna(df_env["항목"])
    df_env["항목"] = pd.Categorical(df_env["항목"], categories=["환기온도", "환기습도"], ordered=True)
    df_env = df_env.sort_values(["항목","연도"]).reset_index(drop=True)
    df_env.index = range(1, len(df_env) + 1)
    df_env.index.name = "No"
    show_styled_dataframe(df_env, "🌡️💧 **연도별 환기온습도 요약**")


    # ✅ 절기별 요약
    df_coil = df_summary[~df_summary["항목명"].isin(["환기온도", "환기습도"])].copy()
    df_coil = df_coil.rename(columns={"항목명": "항목"})

    # 절기별 일수 계산
    df_days = df_coil.groupby(["연도", "절기"])["항목"].count().reset_index(name="일수")

    # merge 해서 절기별 일수 붙이기
    df_coil = df_coil.merge(df_days, on=["연도", "절기"], how="left")

    # 평균 열량을 절기별 일수로 나눔
    df_coil["평균 열량(kWh)"] = df_coil["누적 열량(kWh)"].astype(float) / df_coil["일수"]

    # 표시용 포맷 적용
    df_coil["누적 열량(kWh)"] = df_coil["누적 열량(kWh)"].apply(lambda x: f"{x:,.1f}")
    df_coil["평균 열량(kWh)"] = df_coil["평균 열량(kWh)"].apply(lambda x: f"{x:,.1f}")

    # 필요한 컬럼 정리
    df_coil = df_coil[["항목", "연도", "절기", "누적 열량(kWh)", "평균 열량(kWh)"]]
    df_coil["항목_정규화"] = df_coil["항목"].map(항목명_정규화).fillna(df_coil["항목"])
    df_coil = df_coil.sort_values(["항목", "연도", "절기"]).reset_index(drop=True)
    df_coil.index = range(1, len(df_coil) + 1)
    df_coil.index.name = "No"
    show_styled_dataframe(df_coil, "📌 **절기별 항목별 열량 및 평균 개도율**")

    # ✅ 절기별 환기온습도 요약
    df_env_season = df_summary[df_summary["항목명"].isin(["환기온도", "환기습도"])].copy()
    df_env_season = df_env_season.rename(columns={"항목명": "항목"})
    df_env_season = df_env_season[["항목", "연도", "절기", "평균값"]].copy()
    df_env_season["평균값"] = df_env_season.apply(
        lambda row: f"{row['평균값']:.2f} ℃" if row["항목"] == "환기온도" else f"{row['평균값']:.2f} %", axis=1
    )
    df_env_season["항목_정규화"] = df_env_season["항목"].map(항목명_정규화).fillna(df_env_season["항목"])
    df_env_season["항목"] = pd.Categorical(df_env_season["항목"], categories=["환기온도", "환기습도"], ordered=True)
    df_env_season = df_env_season.sort_values(["항목", "연도", "절기"]).reset_index(drop=True)
    df_env_season.index = range(1, len(df_env_season) + 1)
    df_env_season.index.name = "No"
    show_styled_dataframe(df_env_season, "🌡️💧 **절기별 환기온습도 요약**")



    # ---- 월별 개도율·kWh·비용 (parquet + [선택] RAW 개도율) ----
    coil_items = ["CCV","PC_CCV","HCV","DH_HCV"]

    # 1) parquet(final_analysis)에서 월별 kWh/비용 집계
    coil_items = ["CCV","PC_CCV","HCV","DH_HCV"]
    cols = []
    for h in coil_items:
        for prefix in ["kWh","비용(원)"]:
            col = f"{prefix}_{h}"
            if col in all_df.columns:
                cols.append(col)

    df_ahu_final = all_df[
        (all_df["공조기"] == 선택공조기)
        & (all_df["datetime"] >= 시작)
        & (all_df["datetime"] < 종료)
    ][["datetime"]+cols].copy()

    df_ahu_final = df_ahu_final.melt(id_vars=["datetime"], var_name="지표", value_name="값")
    df_ahu_final["항목명"] = df_ahu_final["지표"].str.split("_").str[-1]
    df_ahu_final["지표타입"] = df_ahu_final["지표"].str.split("_").str[0]  # kWh or 비용(원)


    if df_ahu_final.empty:
        월별_개도율_kWh_비용_표 = pd.DataFrame(columns=["월","항목명","kWh","비용(원)","평균 개도율(%)"])
    else:
        df_ahu_final["월"] = df_ahu_final["datetime"].dt.to_period("M")

        # 지표타입(kWh / 비용(원))을 칼럼으로 피벗해서 합산
        base_monthly = (
            df_ahu_final
            .pivot_table(
                index=["월","항목명"],
                columns="지표타입",      # <- 'kWh', '비용(원)' 값이 들어있음
                values="값",
                aggfunc="sum",
            )
            .reset_index()
        )

        # 피벗 후 컬럼이 없을 수 있으니 방어적으로 보장
        if "kWh" not in base_monthly.columns:
            base_monthly["kWh"] = np.nan
        if "비용(원)" not in base_monthly.columns:
            base_monthly["비용(원)"] = np.nan


        # 2) [선택] RAW(detail parquet)에서 '평균 개도율(%)'만 가중평균으로 계산해서 병합
        raw = load_ahu_detail_by_mode(선택공조기, mode)  # detail parquet(원시시계열) 있으면 사용
        if raw is not None and not raw.empty:
            raw = raw[
                (raw["항목명"].isin(coil_items))
                & (raw["datetime"] >= 시작)
                & (raw["datetime"] < 종료)
            ].copy()
            raw = raw.sort_values("datetime")
            raw["dt_h"] = raw["datetime"].diff().dt.total_seconds()/3600
            raw = raw[(raw["dt_h"] > 0) & (raw["dt_h"] <= 12)].copy()
            if not raw.empty:
                raw["월"] = raw["datetime"].dt.to_period("M")
                # 시간가중 평균 개도율
                wavg = (
                    raw.groupby(["월","항목명"])
                    .apply(lambda g: np.average(g["값"], weights=g["dt_h"]))
                    .reset_index(name="평균 개도율(%)")
                )
                월별_개도율_kWh_비용_표 = base_monthly.merge(wavg, on=["월","항목명"], how="left")
            else:
                월별_개도율_kWh_비용_표 = base_monthly.assign(**{"평균 개도율(%)": np.nan})
        else:
            월별_개도율_kWh_비용_표 = base_monthly.assign(**{"평균 개도율(%)": np.nan})

    # 보기 좋게 월을 문자열로
    월별_개도율_kWh_비용_표["월"] = 월별_개도율_kWh_비용_표["월"].astype(str)


    # 🟩 월별 환기온도/외기온도 평균
    raw = load_ahu_detail_by_mode(선택공조기, mode)
    if raw is not None and not raw.empty:
        df_rat = raw[raw["항목명"] == "RAT"].copy()
        if not df_rat.empty:
            df_rat["월"] = df_rat["datetime"].dt.to_period("M")
            월별_환기온도 = df_rat.groupby("월")["값"].mean().reset_index(name="환기온도 평균(°C)")
        else:
            월별_환기온도 = pd.DataFrame(columns=["월","환기온도 평균(°C)"])
    else:
        월별_환기온도 = pd.DataFrame(columns=["월","환기온도 평균(°C)"])

    # 🟩 월별 환기습도/외기습도 평균
    if raw is not None and not raw.empty:
        df_rah = raw[raw["항목명"] == "RAH"].copy()
        if not df_rah.empty:
            df_rah["월"] = df_rah["datetime"].dt.to_period("M")
            월별_환기습도 = df_rah.groupby("월")["값"].mean().reset_index(name="환기습도 평균(%)")
        else:
            월별_환기습도 = pd.DataFrame(columns=["월","환기습도 평균(%)"])
    else:
        월별_환기습도 = pd.DataFrame(columns=["월","환기습도 평균(%)"])

    # 🟩 외기 데이터 처리
    if not 외기df_hourly.empty:
        _oa = 외기df_hourly.copy()
        _oa["월"] = _oa["datetime"].dt.to_period("M")
        월별_외기온도 = _oa.groupby("월")["외기온도"].mean().reset_index(name="외기온도 평균(°C)")
        월별_외기습도 = _oa.groupby("월")["외기습도"].mean().reset_index(name="외기습도 평균(%)")
    else:
        월별_외기온도 = pd.DataFrame(columns=["월","외기온도 평균(°C)"])
        월별_외기습도 = pd.DataFrame(columns=["월","외기습도 평균(%)"])

    # 🟩 병합: 환기 ↔ 외기
    월별_환기온도_표 = pd.merge(월별_환기온도, 월별_외기온도, on="월", how="outer")
    월별_환기습도_표 = pd.merge(월별_환기습도, 월별_외기습도, on="월", how="outer")



    # 🟩 개별 항목 시각화
    for 선택항목 in ["CCV", "PC_CCV", "HCV", "DH_HCV", "RAT", "RAH"]:
        if raw is not None and not raw.empty:
            df_selected = raw[raw["항목명"] == 선택항목].copy()
        else:
            df_selected = pd.DataFrame()

        if df_selected.empty:
            continue

        # 👉 여기서 그래프 처리 (기존 코드 유지)


        if 선택항목 in ["CCV", "PC_CCV"]:
            y_col = f"{선택항목}_kWh"
            항목_출력명 = 항목명_한글.get(선택항목, 선택항목)
            title = f"❄️ 일일 냉수 에너지 사용량 ({항목_출력명})"
            mx = 항목_열량맵핑[선택항목].get(ahu, 0)

            # ✅ 열량 계산 공통 처리
            df_selected["시간간격"] = df_selected["datetime"].diff().dt.total_seconds().div(3600).fillna(0)
            df_selected = df_selected[(df_selected["시간간격"] > 0) & (df_selected["시간간격"] <= 12)].copy()
            df_selected[y_col] = (
                df_selected["값"].shift(1).add(df_selected["값"]).div(2)
                * mx * df_selected["시간간격"] / 100 / 860
            )
            df_selected["날짜"] = df_selected["datetime"].dt.date
            일별_집계 = df_selected.groupby("날짜")[y_col].sum().reset_index()
            일별_집계["공조기"] = ahu

            일별_집계["연도"] = pd.to_datetime(일별_집계["날짜"]).dt.year
            일별_집계["절기"] = pd.to_datetime(일별_집계["날짜"]).apply(절기_분류)
            일별_집계["월일"] = pd.to_datetime(일별_집계["날짜"]).dt.strftime("%m-%d")
            색상_리스트 = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
            색상_순환기 = itertools.cycle(색상_리스트)
            고유_레전드 = 일별_집계["공조기"] + " | " + 일별_집계["연도"].astype(str)
            color_map = {레전드: next(색상_순환기) for 레전드 in sorted(고유_레전드.unique())}
            st.subheader(title)

            with st.expander(f"{title}_절기별 트렌드", expanded=False):
                draw_season_year_line(
                    일별_집계,
                    y_col=y_col,
                    title=title,
                    평균선_컬럼=y_col,
                    color_map=color_map
                )
            with st.expander(f"⏱️ 개도율 트렌드 ({항목_출력명})", expanded=False):
                draw_overlay_by_shifted_datetime(
                    df=df_selected,
                    y_col="값",
                    title=f"⏱️ 개도율 트렌드 ({항목_출력명})",
                    평균선_컬럼="값"
                )
            

        elif 선택항목 in ["HCV", "DH_HCV"]:
            y_col = f"{선택항목}_kWh"
            항목_출력명 = 항목명_한글.get(선택항목, 선택항목)
            title = f"🔥 일일 증기 에너지 사용량 ({항목_출력명})"
            mx = 항목_열량맵핑[선택항목].get(ahu, 0)

            # ✅ 열량 계산 공통 처리
            df_selected["시간간격"] = df_selected["datetime"].diff().dt.total_seconds().div(3600).fillna(0)
            df_selected = df_selected[(df_selected["시간간격"] > 0) & (df_selected["시간간격"] <= 12)].copy()
            df_selected[y_col] = (
                df_selected["값"].shift(1).add(df_selected["값"]).div(2)
                * mx * df_selected["시간간격"] / 100 / 860
            )
            df_selected["날짜"] = df_selected["datetime"].dt.date
            일별_집계 = df_selected.groupby("날짜")[y_col].sum().reset_index()
            일별_집계["공조기"] = ahu

            일별_집계["연도"] = pd.to_datetime(일별_집계["날짜"]).dt.year
            일별_집계["절기"] = pd.to_datetime(일별_집계["날짜"]).apply(절기_분류)
            일별_집계["월일"] = pd.to_datetime(일별_집계["날짜"]).dt.strftime("%m-%d")
            색상_리스트 = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
            색상_순환기 = itertools.cycle(색상_리스트)
            고유_레전드 = 일별_집계["공조기"] + " | " + 일별_집계["연도"].astype(str)
            color_map = {레전드: next(색상_순환기) for 레전드 in sorted(고유_레전드.unique())}

            st.subheader(title)

            with st.expander(f"{title}_절기별 트렌드", expanded=False):
                draw_season_year_line(
                    일별_집계,
                    y_col=y_col,
                    title=title + " (절기별 트렌드)",
                    평균선_컬럼=y_col,
                    color_map=color_map
                )

            with st.expander(f"⏱️ 개도율 트렌드 ({항목_출력명})", expanded=False):
                draw_overlay_by_shifted_datetime(
                    df=df_selected,
                    y_col="값",
                    title=f"⏱️ 개도율 트렌드 ({항목_출력명})",
                    평균선_컬럼="값",
                    color_map=color_map
                )
            
        elif 선택항목 in ["RAT", "RAH"]:
            if "환기_외기_요약_출력됨" not in st.session_state:
                st.markdown("### 🌡️💧 환기온습도·외기온습도 트렌드")
                st.session_state["환기_외기_요약_출력됨"] = True

            if not 외기df_hourly.empty:
                label = "환기온도" if 선택항목 == "RAT" else "환기습도"
                with st.expander(f"📈 {label} 및 외기 비교", expanded=False):
                    

                    for 연도 in sorted(df_selected["datetime"].dt.year.unique()):
                        df_year  = df_selected[df_selected["datetime"].dt.year == 연도].copy()
                        ext_year = 외기df_hourly[외기df_hourly["datetime"].dt.year == 연도].copy()

                        if df_year.empty:
                            continue

                        fig = go.Figure()

                        # ✅ 외기값 처리
                        ext_year = ext_year.sort_values("datetime")
                        ext_year["시간차"] = ext_year["datetime"].diff().dt.total_seconds().div(60)
                        ext_year["gap_group"] = (ext_year["시간차"] > 300).cumsum()

                        ext_legend_shown = False
                        if 선택항목 == "RAT":
                            for _, g in ext_year.groupby("gap_group"):
                                if g.empty: 
                                    continue
                                fig.add_trace(go.Scatter(
                                    x=g["datetime"], y=g["외기온도"],
                                    mode="lines", name="외기온도",
                                    line=dict(color="gray"),
                                    connectgaps=False,
                                    showlegend=not ext_legend_shown,  # ← 첫 그룹만 범례 표시
                                    legendgroup="외기온도"
                                ))
                                ext_legend_shown = True
                        else:
                            for _, g in ext_year.groupby("gap_group"):
                                if g.empty:
                                    continue
                                fig.add_trace(go.Scatter(
                                    x=g["datetime"], y=g["외기습도"],
                                    mode="lines", name="외기습도",
                                    line=dict(color="gray"),
                                    connectgaps=False,
                                    showlegend=not ext_legend_shown,  # ← 첫 그룹만 범례 표시
                                    legendgroup="외기습도"
                                ))
                                ext_legend_shown = True

                        # ✅ 환기값 처리
                        df_year = df_year.sort_values("datetime")
                        df_year["시간차"] = df_year["datetime"].diff().dt.total_seconds().div(60)
                        df_year["gap_group"] = (df_year["시간차"] > 300).cumsum()

                        vent_name = "환기온도" if 선택항목 == "RAT" else "환기습도"
                        vent_legend_shown = False  # ← 추가
                        for _, g in df_year.groupby("gap_group"):
                            if g.empty:
                                continue
                            fig.add_trace(go.Scatter(
                                x=g["datetime"], y=g["값"],
                                mode="lines", name=vent_name,
                                line=dict(color="blue"),
                                connectgaps=False,
                                showlegend=not vent_legend_shown,  # ← 첫 그룹만 범례 표시
                                legendgroup="환기"
                            ))
                            vent_legend_shown = True


                        # ✅ 기준선 및 밴드
                        if 선택항목 == "RAT":
                            if ahu in BAND_RANGES_RAT:
                                for ymin, ymax in BAND_RANGES_RAT[ahu]:
                                    fig = add_band(fig, ymin, ymax, color="orange", label="경고구간")
                            if ahu in AHU_RAT_LIMITS:
                                fig.add_hline(y=AHU_RAT_LIMITS[ahu][0], line_dash="dot", line_color="red",
                                                annotation_text=f"{AHU_RAT_LIMITS[ahu][0]}°C", annotation_position="top left")
                                fig.add_hline(y=AHU_RAT_LIMITS[ahu][1], line_dash="dot", line_color="red",
                                                annotation_text=f"{AHU_RAT_LIMITS[ahu][1]}°C", annotation_position="top left")
                            fig.update_layout(
                                title=f"{연도}년 환기온도 및 외기온도",
                                xaxis_title="날짜", yaxis_title="온도", legend=dict(y=1, x=1.05)
                            )
                        else:
                            if ahu in BAND_RANGES_RAH:
                                for ymin, ymax in BAND_RANGES_RAH[ahu]:
                                    fig = add_band(fig, ymin, ymax, color="orange", label="경고구간")
                            if ahu in AHU_RAH_LIMITS:
                                y_limit = AHU_RAH_LIMITS[ahu][0]
                                fig.add_hline(y=y_limit, line_dash="dot", line_color="red",
                                                annotation_text=f"{y_limit}%", annotation_position="top left")
                            fig.update_layout(
                                title=f"{연도}년 환기습도 및 외기습도",
                                xaxis_title="날짜", yaxis_title="습도", legend=dict(y=1, x=1.05)
                            )

                        # ✅ 기본 설정
                        fig.update_xaxes(type="date", showgrid=True, tickformat="%m-%d\n%H:%M")
                        fig.update_yaxes(showline=True, linecolor="black")
                        st.plotly_chart(fig, use_container_width=True)


t_total_end = time.time()
if st.session_state.get("t_total_start") is not None:
    st.success(f"🧮 총 분석 완료 시간: {t_total_end - st.session_state['t_total_start']:.1f}초")

else:
    st.info("데이터가 준비되지 않았습니다. 잠시 후 '데이터 강제 재분석' 버튼을 클릭하거나, history 폴더의 CSV 파일을 확인해주세요.")
