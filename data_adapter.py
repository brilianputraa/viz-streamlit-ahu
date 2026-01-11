"""
Unified data access layer for viz-streamlit-ahu.

Provides abstraction over data sources (parquet files vs database).
"""
import os
import sys
import importlib.util
from pathlib import Path
import streamlit as st
from enum import Enum
from typing import Literal, Optional
import pandas as pd
import psycopg2
import warnings
import time
from datetime import datetime  # [추가됨] 안전한 기본 날짜 범위 fallback(today) 계산용

# Import existing parquet loader functions
from loader import (
    load_final_results as load_parquet_final_results,
    load_ahu_detail as load_parquet_ahu_detail,
    load_oa_results as load_parquet_oa_results,
    load_oa_daily as load_parquet_oa_daily
)
from db_config import get_database_connection_config

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

# [추가됨] DB 재시작/복구모드 등 일시 장애 대비 재시도 연결
def _connect_with_retry(config, attempts: int = 6, base_delay_s: float = 0.5):
    """
    Connect to Postgres with small retries for transient states like recovery mode.

    This helps when the DB restarts (e.g., after OOM) and briefly rejects logins.
    """
    last_exc = None
    for i in range(attempts):
        try:
            cfg = dict(config)
            cfg.setdefault("connect_timeout", 5)
            return psycopg2.connect(**cfg)
        except psycopg2.OperationalError as e:
            last_exc = e
            msg = str(e).lower()
            transient = (
                "recovery mode" in msg
                or "server closed the connection unexpectedly" in msg
                or "could not connect to server" in msg
                or "connection refused" in msg
            )
            if not transient or i == attempts - 1:
                raise
            time.sleep(base_delay_s * (2 ** i))
    raise last_exc  # pragma: no cover


# [추가됨] DB에 존재하는 energy_readings 최신 날짜 조회 (기본 날짜 범위 자동 세팅용)
@st.cache_data(ttl=300)
def get_latest_energy_date(mode: "DataAccessMode" = None):
    if mode is None:
        mode = DataAccessMode.PARQUET

    if mode != DataAccessMode.DATABASE:
        return None

    try:
        config = get_database_connection_config()
        conn = _connect_with_retry(config)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT MAX(timestamp)::date FROM ahu_data.energy_readings WHERE timestamp IS NOT NULL"
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.close()
    except Exception:
        return None


# [추가됨] DB에 존재하는 센서(ahu_readings_staging) 최신 날짜 조회
@st.cache_data(ttl=300)
def get_latest_sensor_date(mode: "DataAccessMode" = None):
    if mode is None:
        mode = DataAccessMode.PARQUET

    if mode != DataAccessMode.DATABASE:
        return None

    try:
        config = get_database_connection_config()
        conn = _connect_with_retry(config)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT MAX(timestamp)::date FROM ahu_data.ahu_readings_staging WHERE timestamp IS NOT NULL"
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.close()
    except Exception:
        return None


# [추가됨] DB에 존재하는 외기(outdoor_weather) 최신 날짜 조회
@st.cache_data(ttl=300)
def get_latest_oa_date(mode: "DataAccessMode" = None):
    if mode is None:
        mode = DataAccessMode.PARQUET

    if mode != DataAccessMode.DATABASE:
        return None

    try:
        config = get_database_connection_config()
        conn = _connect_with_retry(config)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT MAX(timestamp)::date FROM ahu_data.outdoor_weather WHERE timestamp IS NOT NULL"
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.close()
    except Exception:
        return None


def _default_range_from_latest(latest, days: int = 30):
    if latest is None:
        return None, None
    safe_end = str(latest)
    safe_start = str(latest - pd.Timedelta(days=days))
    return safe_start, safe_end


def _month_windows(start_date: str, end_date: str):
    """
    Split an inclusive [start_date, end_date] range into month-sized windows.

    Returns list of (start_str, end_str) date strings (YYYY-MM-DD), inclusive end.
    """
    start = pd.to_datetime(start_date, errors="raise").normalize()
    end = pd.to_datetime(end_date, errors="raise").normalize()
    if start > end:
        return []

    windows = []
    current = start
    while current <= end:
        month_end = (current + pd.offsets.MonthEnd(0)).normalize()
        window_end = month_end if month_end <= end else end
        windows.append((str(current.date()), str(window_end.date())))
        current = (month_end + pd.Timedelta(days=1)).normalize()
    return windows

# [수정됨] ahu_query_lib 자동 로드 (ahu-backend-server 경로 자동 감지)
# Modified: PYTHONPATH가 없어도 ahu-backend-server 폴더에서 ahu_query_lib를 찾도록 보완
def ensure_ahu_query_lib(import_module: bool = True):
    """
    Import and return `ahu_query_lib`, attempting to auto-discover the repo path.

    Supports:
      - `ahu_query_lib` already installed in the environment
      - Monorepo layout: `viz-streamlit-ahu/` is inside `ahu-backend-server/`
      - Sibling layout: `viz-streamlit-ahu/` next to `ahu-backend-server/`

    You can also explicitly set `AHU_BACKEND_SERVER_PATH` (one or more paths,
    separated by `os.pathsep`) to point at the `ahu-backend-server` directory.
    """
    if not import_module:
        if importlib.util.find_spec("ahu_query_lib") is not None:
            return importlib.util.find_spec("ahu_query_lib")
    else:
        try:
            import ahu_query_lib as aql
            return aql
        except ImportError:
            pass

    candidates = []

    env_paths = os.getenv("AHU_BACKEND_SERVER_PATH", "")
    for raw in env_paths.split(os.pathsep):
        raw = (raw or "").strip()
        if raw:
            candidates.append(Path(raw).expanduser())

    base_dir = Path(__file__).resolve().parent
    for start in (Path.cwd(), base_dir):
        try:
            start = start.resolve()
        except Exception:
            continue
        for parent in [start, *start.parents][:6]:
            candidates.append(parent)
            candidates.append(parent / "ahu-backend-server")

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)

        if not (candidate / "ahu_query_lib").is_dir():
            continue

        if key not in sys.path:
            sys.path.insert(0, key)

        if not import_module:
            spec = importlib.util.find_spec("ahu_query_lib")
            if spec is not None:
                return spec
            continue

        import ahu_query_lib as aql
        return aql

    return None

# Enum for data access modes
class DataAccessMode(Enum):
    """Data source mode options."""
    PARQUET = "parquet"
    DATABASE = "database"


@st.cache_data(ttl=300)  # 5 minute cache
def load_final_results(
    mode: DataAccessMode = DataAccessMode.PARQUET,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load final aggregated results from selected data source.

    Args:
        mode: Data source mode (parquet or database)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        DataFrame with final energy data (kWh, costs, etc.)

    Example:
        >>> df = load_final_results(mode=DataAccessMode.DATABASE)
        >>> print(df.columns)
    """
    if mode == DataAccessMode.PARQUET:
        return load_parquet_final_results()
    elif mode == DataAccessMode.DATABASE:
        try:
            # [추가됨] 기간 미지정 시 DB 최신 날짜 기준으로 안전한 기본 조회 범위 사용
            if start_date is None or end_date is None:
                latest = (
                    get_latest_energy_date(mode=DataAccessMode.DATABASE)
                    or get_latest_oa_date(mode=DataAccessMode.DATABASE)
                    or get_latest_sensor_date(mode=DataAccessMode.DATABASE)
                )
                safe_start, safe_end = _default_range_from_latest(latest, days=30)
                # [추가됨] 어떤 테이블에서도 최신 날짜를 못 찾으면(=DB 장애/빈 DB) 최후 수단으로 today 사용
                if safe_start is None or safe_end is None:
                    today = datetime.now().date()
                    safe_start, safe_end = _default_range_from_latest(today, days=30)
                start_date = start_date or safe_start
                end_date = end_date or safe_end

            # [추가됨] ahu_query_lib를 통해 조회 (내부에서 cagg 우선 사용 + 필요시 raw fallback)
            aql = ensure_ahu_query_lib()
            if not aql:
                st.error("ahu_query_lib not available. Set PYTHONPATH or AHU_BACKEND_SERVER_PATH.")
                return pd.DataFrame()

            df = aql.fetch_energy_consumption_cost(
                ahu_id=None,
                start_period=start_date,
                end_period=end_date,
                data_period="daily",
                detailed_cost=True,
                include_weather=True,
                include_granular=True,
                include_korean=True,
            )

            # Normalize period column returned by ahu_query_lib to datetime for app compatibility.
            if not df.empty and "datetime" not in df.columns and "period" in df.columns:
                dt = pd.to_datetime(df["period"], errors="coerce")
                if getattr(dt.dt, "tz", None) is not None:
                    dt = dt.dt.tz_localize(None)
                df["datetime"] = dt
            # Map DB column name to Korean label expected by viz app.
            if not df.empty and "공조기" not in df.columns and "ahu_id" in df.columns:
                df = df.rename(columns={"ahu_id": "공조기"})
            return df
        except Exception as e:
            # Fallback: try ahu_query_lib path (kept for compatibility)
            try:
                aql = ensure_ahu_query_lib()
                if not aql:
                    st.error("ahu_query_lib not available. Set PYTHONPATH or AHU_BACKEND_SERVER_PATH.")
                    return pd.DataFrame()

                # [추가됨] fallback에서도 랜덤/고정 날짜가 아닌 DB 최신 날짜 기반 기본 범위 사용
                if start_date is None or end_date is None:
                    latest = (
                        get_latest_energy_date(mode=DataAccessMode.DATABASE)
                        or get_latest_oa_date(mode=DataAccessMode.DATABASE)
                        or get_latest_sensor_date(mode=DataAccessMode.DATABASE)
                    )
                    safe_start, safe_end = _default_range_from_latest(latest, days=30)
                    if safe_start is None or safe_end is None:
                        today = datetime.now().date()
                        safe_start, safe_end = _default_range_from_latest(today, days=30)
                    start_date = start_date or safe_start
                    end_date = end_date or safe_end

                df = aql.fetch_energy_consumption_cost(
                    ahu_id=None,  # All AHUs
                    start_period=start_date,
                    end_period=end_date,
                    data_period="daily",
                    detailed_cost=True,
                    include_weather=True,
                    include_granular=True,
                    include_korean=True,
                )
                if not df.empty and "datetime" not in df.columns and "period" in df.columns:
                    dt = pd.to_datetime(df["period"], errors="coerce")
                    if getattr(dt.dt, "tz", None) is not None:
                        dt = dt.dt.tz_localize(None)
                    df["datetime"] = dt
                if not df.empty and "공조기" not in df.columns and "ahu_id" in df.columns:
                    df = df.rename(columns={"ahu_id": "공조기"})
                return df
            except Exception as e2:
                st.error(f"Database error: {e}")
                st.error(f"Fallback error: {e2}")
                return pd.DataFrame()
    else:
        raise ValueError(f"Invalid mode: {mode}")


@st.cache_data(ttl=300)
def load_ahu_detail(
    ahu_name: str,
    mode: DataAccessMode = DataAccessMode.PARQUET,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load detailed sensor data for a specific AHU.

    Args:
        ahu_name: AHU identifier (e.g., 'AHU01')
        mode: Data source mode
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with detailed sensor readings in long format:
        [datetime, 공조기, 항목명, 값]
    """
    if mode == DataAccessMode.PARQUET:
        return load_parquet_ahu_detail(ahu_name)
    elif mode == DataAccessMode.DATABASE:
        try:
            aql = ensure_ahu_query_lib()
            if not aql:
                st.error("ahu_query_lib not available. Set PYTHONPATH or AHU_BACKEND_SERVER_PATH.")
                return pd.DataFrame()

            # [추가됨] 센서 데이터는 매우 커서 기본값을 "최근 30일(최신 센서 날짜 기준)"로 제한
            if start_date is None or end_date is None:
                latest = get_latest_sensor_date(mode=DataAccessMode.DATABASE) or get_latest_energy_date(
                    mode=DataAccessMode.DATABASE
                )
                safe_start, safe_end = _default_range_from_latest(latest, days=30)
                if safe_start is None or safe_end is None:
                    today = datetime.now().date()
                    safe_start, safe_end = _default_range_from_latest(today, days=30)
                start_date = start_date or safe_start
                end_date = end_date or safe_end
            # Fetch from staging table via ahu_query_lib
            df = aql.fetch_sensor_data(
                ahu_id=ahu_name,
                start_date=start_date,
                end_date=end_date,
                aggregate="raw"
            )

            if df.empty:
                return pd.DataFrame(columns=['datetime', '공조기', '항목명', '값'])

            # Transform to long format expected by visualization
            # Columns from ahu_query_lib: timestamp, ahu_id, CCV, HCV, SFST, etc.
            id_vars = ['timestamp', 'ahu_id']
            # Find value columns (exclude timestamp, ahu_id, outdoor_temperature, outdoor_humidity)
            value_cols = [col for col in df.columns
                         if col not in ['timestamp', 'ahu_id', 'outdoor_temperature', 'outdoor_humidity']]

            df_long = df.melt(
                id_vars=id_vars,
                value_vars=value_cols,
                var_name='항목명',
                value_name='값'
            )

            # Drop rows with null values
            df_long = df_long.dropna(subset=['값'])

            # Convert '값' column to numeric type
            df_long['값'] = pd.to_numeric(df_long['값'], errors='coerce')

            # Rename columns to match expected format
            df_long['공조기'] = df_long['ahu_id']
            df_long['datetime'] = pd.to_datetime(df_long['timestamp']).dt.tz_localize(None)

            return df_long[['datetime', '공조기', '항목명', '값']]

        except Exception as e:
            st.error(f"Database error loading AHU detail: {e}")
            return pd.DataFrame()
    else:
        raise ValueError(f"Invalid mode: {mode}")


@st.cache_data(ttl=300)
def load_oa_data(
    mode: DataAccessMode = DataAccessMode.PARQUET,
    daily: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load outdoor air data.

    Args:
        mode: Data source mode
        daily: If True, return daily averages; if False, return raw data
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with outdoor temperature and humidity.
        Parquet mode: [datetime, 외기온도, 외기습도]
        Database mode: [datetime, 외기온도, 외기습도]
    """
    if mode == DataAccessMode.PARQUET:
        if daily:
            return load_parquet_oa_daily()
        else:
            return load_parquet_oa_results()
    elif mode == DataAccessMode.DATABASE:
        try:
            # Query outdoor_weather table directly
            config = get_database_connection_config()
            conn = _connect_with_retry(config)

            # [추가됨] 기간 미지정 시 energy_readings 최신 날짜 기준으로 안전한 기본 조회 범위 사용
            if start_date is None or end_date is None:
                latest = (
                    get_latest_oa_date(mode=DataAccessMode.DATABASE)
                    or get_latest_energy_date(mode=DataAccessMode.DATABASE)
                    or get_latest_sensor_date(mode=DataAccessMode.DATABASE)
                )
                safe_start, safe_end = _default_range_from_latest(latest, days=30)
                if safe_start is None or safe_end is None:
                    today = datetime.now().date()
                    safe_start, safe_end = _default_range_from_latest(today, days=30)
                start_date = start_date or safe_start
                end_date = end_date or safe_end

            if daily:
                # Get daily averages
                query = '''
                    SELECT
                        DATE_TRUNC('day', timestamp)::date as date,
                        AVG(outdoor_temperature) as outdoor_temperature,
                        AVG(outdoor_humidity) as outdoor_humidity
                    FROM ahu_data.outdoor_weather
                    WHERE timestamp IS NOT NULL
                '''

                params = []
                if start_date:
                    query += " AND timestamp >= %s"
                    params.append(start_date)
                if end_date:
                    query += " AND timestamp <= %s"
                    params.append(end_date)

                query += " GROUP BY DATE_TRUNC('day', timestamp)::date ORDER BY date"

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="pandas only supports SQLAlchemy connectable.*",
                        category=UserWarning,
                    )
                    df = pd.read_sql(query, conn, params=params)

                if not df.empty:
                    df['datetime'] = pd.to_datetime(df['date'])
                    df = df.drop(columns=['date'])

            else:
                # Get raw data
                query = '''
                    SELECT timestamp, outdoor_temperature, outdoor_humidity
                    FROM ahu_data.outdoor_weather
                    WHERE timestamp IS NOT NULL
                '''

                params = []
                if start_date:
                    query += " AND timestamp >= %s"
                    params.append(start_date)
                if end_date:
                    query += " AND timestamp <= %s"
                    params.append(end_date)

                query += " ORDER BY timestamp"

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="pandas only supports SQLAlchemy connectable.*",
                        category=UserWarning
                    )
                    df = pd.read_sql(query, conn, params=params)
                if not df.empty:
                    df['datetime'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

            conn.close()

            if df.empty:
                return pd.DataFrame(columns=['datetime', '외기온도', '외기습도'])

            # Rename columns to Korean names expected by visualization
            df = df.rename(columns={
                'outdoor_temperature': '외기온도',
                'outdoor_humidity': '외기습도'
            })

            return df[['datetime', '외기온도', '외기습도']]

        except Exception as e:
            st.error(f"Database error loading OA data: {e}")
            import traceback
            st.error(traceback.format_exc())
            return pd.DataFrame()
    else:
        raise ValueError(f"Invalid mode: {mode}")


def get_available_ahu_list(
    mode: DataAccessMode = DataAccessMode.PARQUET
) -> list:
    """
    Get list of available AHU identifiers.

    Args:
        mode: Data source mode

    Returns:
        List of AHU IDs (e.g., ['AHU01', 'AHU02', ...])
    """
    if mode == DataAccessMode.PARQUET:
        # From common.py hardcoded list
        return [
            "AHU01", "AHU02", "AHU03", "AHU04", "AHU05", "AHU06", "AHU07",
            "AHU08", "AHU09", "AHU10", "AHU11", "AHU12", "AHU13", "AHU14",
            "AHU15", "AHU16", "AHU17", "AHU18", "AHU19", "AHU20", "AHU21",
            "AHU22", "AHU23", "AHU24", "AHU25", "AHU26", "AHU32", "AHU33",
            "AHU36", "AHU37", "AHU40", "AHU41", "AHU42", "AHU43", "AHU44"
        ]
    elif mode == DataAccessMode.DATABASE:
        try:
            aql = ensure_ahu_query_lib()
            if not aql:
                return get_available_ahu_list(DataAccessMode.PARQUET)
            metadata = aql.fetch_all_ahu_metadata()
            return sorted(metadata.keys())
        except Exception:
            # Fallback to default list
            return get_available_ahu_list(DataAccessMode.PARQUET)
    else:
        raise ValueError(f"Invalid mode: {mode}")
