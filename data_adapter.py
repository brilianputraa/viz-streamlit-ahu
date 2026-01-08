"""
Unified data access layer for viz-streamlit-ahu.

Provides abstraction over data sources (parquet files vs database).
"""
import streamlit as st
from enum import Enum
from typing import Literal, Optional
import pandas as pd

# Import existing parquet loader functions
from loader import (
    load_final_results as load_parquet_final_results,
    load_ahu_detail as load_parquet_ahu_detail,
    load_oa_results as load_parquet_oa_results,
    load_oa_daily as load_parquet_oa_daily
)

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
        # Import here to avoid issues when ahu_query_lib not available
        try:
            import ahu_query_lib as aql
            return aql.fetch_energy_consumption_cost(
                ahu_id=None,  # All AHUs
                start_period=start_date or "2021-01-01",
                end_period=end_date or "2025-12-31",
                data_period="daily",
                detailed_cost=True,
                include_weather=True,
                include_granular=True,
                include_korean=True
            )
        except ImportError:
            st.error("ahu_query_lib not available. Install with: pip install ahu_query_lib")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Database error: {e}")
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
        DataFrame with detailed sensor readings
    """
    if mode == DataAccessMode.PARQUET:
        return load_parquet_ahu_detail(ahu_name)
    elif mode == DataAccessMode.DATABASE:
        try:
            import ahu_query_lib as aql
            return aql.fetch_sensor_data(
                ahu_id=ahu_name,
                start_date=start_date or "2021-01-01",
                end_date=end_date or "2025-12-31",
                aggregate="raw"
            )
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
        DataFrame with outdoor temperature and humidity
    """
    if mode == DataAccessMode.PARQUET:
        if daily:
            return load_parquet_oa_daily()
        else:
            return load_parquet_oa_results()
    elif mode == DataAccessMode.DATABASE:
        try:
            import ahu_query_lib as aql
            # Use AHU01's outdoor sensor (same for all AHUs)
            return aql.fetch_sensor_data(
                ahu_id="AHU01",
                start_date=start_date or "2021-01-01",
                end_date=end_date or "2025-12-31",
                parameters=["OA_T", "OA_H"],
                aggregate="daily" if daily else "raw"
            )
        except Exception as e:
            st.error(f"Database error loading OA data: {e}")
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
            import ahu_query_lib as aql
            metadata = aql.fetch_all_ahu_metadata()
            return sorted(metadata.keys())
        except Exception:
            # Fallback to default list
            return get_available_ahu_list(DataAccessMode.PARQUET)
    else:
        raise ValueError(f"Invalid mode: {mode}")
