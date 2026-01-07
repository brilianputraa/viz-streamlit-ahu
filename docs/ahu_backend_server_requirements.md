# ahu-backend-server: Required Additions for Viz Integration

**Date:** 2026-01-07
**Repository:** `/Users/brilian/ahu-backend-server`
**Purpose:** List ALL features to add to ahu_query_lib for seamless Streamlit viz integration

---

## Overview

The `ahu-backend-server` repository contains `ahu_query_lib` - the database query library used by the visualization app. This document lists **all required additions** to make the library work properly with the Streamlit viz program.

**Current Status:**
- ❌ Library exists but returns empty data (queries empty `daily_energy_summary` table)
- ❌ Column name mismatches cause query errors
- ❌ Missing granular cost breakdown (CCV, PC_CCV, HCV, DH_HCV)
- ❌ No batch query support (loops 45 AHUs individually)
- ❌ Timezone handling issues
- ❌ Performance problems (no indexes, 5+ second queries)

---

## File Structure Changes

### Files to Create:

```
ahu-backend-server/
└── ahu_query_lib/
    ├── api/
    │   ├── batch.py           # NEW: Batch query functions
    │   └── outdoor_air.py     # NEW: Outdoor air data functions
    ├── cache/
    │   ├── __init__.py
    │   └── query_cache.py     # NEW: Query result caching
    └── utils/
        ├── timezone.py        # NEW: Timezone handling utilities
        └── transformers.py    # NEW: Data format transformers
```

### Files to Modify:

```
ahu-backend-server/
└── ahu_query_lib/
    ├── queries/
    │   └── energy_queries.py  # MODIFY: Fix column names, add granular
    ├── db/
    │   └── connection.py      # MODIFY: Add connection pooling
    └── api/
        └── energy.py           # MODIFY: Add new parameters, batch support
```

---

## Detailed Implementation Requirements

### 1. Fix Column Name Mismatches 🔴 CRITICAL

**File:** `ahu_query_lib/queries/energy_queries.py`

**Current Code (BROKEN):**
```python
# Line ~45:
"AVG(er.outdoor_temperature) as avg_outdoor_temp"
"AVG(er.outdoor_humidity) as avg_outdoor_humidity"
```

**Required Fix:**
```python
# Change to:
"AVG(er.outdoor_temperature) as outdoor_temperature"
"AVG(er.outdoor_humidity) as outdoor_humidity"

# Also update return DataFrame columns:
return pd.DataFrame({
    "period": result["period"],
    "ahu_id": result["ahu_id"],
    "outdoor_temperature": result["outdoor_temperature"],  # Changed
    "outdoor_humidity": result["outdoor_humidity"],          # Changed
    ...
})
```

**Why:** Database columns are `outdoor_temperature` and `outdoor_humidity`, not `avg_outdoor_temp` and `avg_outdoor_humidity`.

---

### 2. Change Query Source Table 🔴 CRITICAL

**File:** `ahu_query_lib/queries/energy_queries.py`

**Current Code (BROKEN):**
```python
# Line ~30:
query = """
SELECT ...
FROM ahu_data.daily_energy_summary  # EMPTY TABLE!
WHERE date >= %s AND date < %s
"""
```

**Required Fix:**
```python
query = """
SELECT
    date_trunc('day', timestamp) as period,
    ahu_id,
    SUM(ccv_cold_water_kwh) as ccv_cold_water_kwh,
    SUM(ac_ccv_cold_water_kwh) as ac_ccv_cold_water_kwh,
    SUM(pc_ccv_cold_water_kwh) as pc_ccv_cold_water_kwh,
    SUM(hcv_steam_kwh) as hcv_steam_kwh,
    SUM(ac_hcv_steam_kwh) as ac_hcv_steam_kwh,
    SUM(dh_hcv_steam_kwh) as dh_hcv_steam_kwh,
    SUM(sf_electricity_kwh) as sf_electricity_kwh,
    SUM(ccv_cold_water_cost_krw) as ccv_cold_water_cost_krw,
    SUM(ac_ccv_cold_water_cost_krw) as ac_ccv_cold_water_cost_krw,
    SUM(pc_ccv_cold_water_cost_krw) as pc_ccv_cold_water_cost_krw,
    SUM(hcv_steam_cost_krw) as hcv_steam_cost_krw,
    SUM(ac_hcv_steam_cost_krw) as ac_hcv_steam_cost_krw,
    SUM(dh_hcv_steam_cost_krw) as dh_hcv_steam_cost_krw,
    SUM(sf_cost_krw) as sf_cost_krw,
    SUM(electricity_cost_krw) as electricity_cost_krw,
    AVG(outdoor_temperature) as outdoor_temperature,
    AVG(outdoor_humidity) as outdoor_humidity
FROM ahu_data.energy_readings  # USE THIS INSTEAD
WHERE timestamp >= %s AND timestamp < %s
  AND ahu_id = %s
GROUP BY date_trunc('day', timestamp), ahu_id
ORDER BY period
"""
```

**Why:** `daily_energy_summary` table is empty (0 rows). All data is in `energy_readings` (815,140 rows of 6-minute data). Need to aggregate on-the-fly with GROUP BY.

---

### 3. Add Granular Cost Breakdown 🔴 CRITICAL

**File:** `ahu_query_lib/api/energy.py`

**Add New Parameter:**
```python
def fetch_energy_consumption_cost(
    ahu_id: str,
    start_period: str,
    end_period: str,
    data_period: str = "daily",
    detailed_cost: bool = False,  # RENAME THIS
    include_weather: bool = False,
    include_granular: bool = False,  # NEW PARAMETER
    include_korean: bool = False    # NEW PARAMETER
) -> pd.DataFrame:
    """
    Fetch energy consumption and cost data.

    Args:
        ahu_id: AHU identifier (e.g., "AHU01")
        start_period: Start date (YYYY-MM-DD)
        end_period: End date (YYYY-MM-DD)
        data_period: "daily", "hourly", or "raw"
        detailed_cost: (DEPRECATED) Use include_granular instead
        include_weather: Include outdoor temperature/humidity
        include_granular: Include CCV/PC_CCV/HCV/DH_HCV breakdown
        include_korean: Include Korean column names (for viz app)

    Returns:
        DataFrame with energy and cost data.

        If include_granular=False:
            Columns: [period, ahu_id, cold_water_energy_kwh,
                     steam_energy_kwh, electricity_energy_kwh, ...]

        If include_granular=True:
            Columns: [period, ahu_id,
                     # kWh values
                     kWh_CCV, kWh_PC_CCV, kWh_HCV, kWh_DH_HCV,
                     kWh_AC_CCV, kWh_AC_HCV,
                     # Cost values (KRW)
                     비용(원)_CCV, 비용(원)_PC_CCV,
                     비용(원)_HCV, 비용(원)_DH_HCV,
                     전력_비용(원), 냉수_비용(원),
                     스팀_비용(원), 총합_비용(원)]
    """
```

**Implementation:**
```python
# After fetching data, transform if include_granular=True:
if include_granular:
    df = _transform_to_granular_format(df, include_korean=include_korean)
return df

def _transform_to_granular_format(df: pd.DataFrame, include_korean: bool = False):
    """Transform summary data to granular CCV/PC_CCV/HCV/DH_HCV format."""
    result = pd.DataFrame()

    # Basic columns
    result["period"] = df["period"]
    result["ahu_id"] = df["ahu_id"]

    # kWh values (map from raw columns)
    result["kWh_CCV"] = df["ccv_cold_water_kwh"] + df["ac_ccv_cold_water_kwh"]
    result["kWh_PC_CCV"] = df["pc_ccv_cold_water_kwh"]
    result["kWh_HCV"] = df["hcv_steam_kwh"] + df["ac_hcv_steam_kwh"]
    result["kWh_DH_HCV"] = df["dh_hcv_steam_kwh"]

    # Cost values
    if include_korean:
        result["비용(원)_CCV"] = df["ccv_cold_water_cost_krw"] + df["ac_ccv_cold_water_cost_krw"]
        result["비용(원)_PC_CCV"] = df["pc_ccv_cold_water_cost_krw"]
        result["비용(원)_HCV"] = df["hcv_steam_cost_krw"] + df["ac_hcv_steam_cost_krw"]
        result["비용(원)_DH_HCV"] = df["dh_hcv_steam_cost_krw"]
        result["전력_비용(원)"] = df["sf_cost_krw"] + df["electricity_cost_krw"]
        result["냉수_비용(원)"] = result["비용(원)_CCV"] + result["비용(원)_PC_CCV"]
        result["스팀_비용(원)"] = result["비용(원)_HCV"] + result["비용(원)_DH_HCV"]
        result["총합_비용(원)"] = (
            result["전력_비용(원)"] +
            result["냉수_비용(원)"] +
            result["스팀_비용(원)"]
        )
    else:
        result["cost_ccv_krw"] = df["ccv_cold_water_cost_krw"] + df["ac_ccv_cold_water_cost_krw"]
        result["cost_pccv_krw"] = df["pc_ccv_cold_water_cost_krw"]
        result["cost_hcv_krw"] = df["hcv_steam_cost_krw"] + df["ac_hcv_steam_cost_krw"]
        result["cost_dhhcv_krw"] = df["dh_hcv_steam_cost_krw"]
        result["electricity_cost_krw"] = df["sf_cost_krw"] + df["electricity_cost_krw"]

    return result
```

---

### 4. Add Batch Query Function 🔴 CRITICAL

**New File:** `ahu_query_lib/api/batch.py`

```python
"""
Batch query functions for fetching data from multiple AHUs efficiently.
"""
from typing import List, Optional, Literal
import pandas as pd
from ahu_query_lib.core.connection import DatabaseManager
from ahu_query_lib.api.energy import fetch_energy_consumption_cost


def fetch_batch_energy_consumption(
    start_period: str,
    end_period: str,
    ahu_list: Optional[List[str]] = None,
    data_period: Literal["daily", "hourly"] = "daily",
    include_granular: bool = False,
    include_korean: bool = False,
    include_weather: bool = False
) -> pd.DataFrame:
    """
    Fetch energy data for multiple AHUs in a SINGLE query.

    This is MUCH faster than looping through individual AHU queries:
    - Looping: 45 queries × 100ms = 4.5 seconds
    - Batch: 1 query × 100ms = 0.1 seconds (45x faster!)

    Args:
        start_period: Start date (YYYY-MM-DD)
        end_period: End date (YYYY-MM-DD)
        ahu_list: List of AHU IDs (e.g., ["AHU01", "AHU02", ...])
                 If None, fetches all available AHUs
        data_period: "daily" or "hourly"
        include_granular: Include CCV/PC_CCV/HCV/DH_HCV breakdown
        include_korean: Include Korean column names
        include_weather: Include outdoor temperature/humidity

    Returns:
        DataFrame with columns:
        - If include_granular=False:
            [period, ahu_id, cold_water_energy_kwh, steam_energy_kwh,
             electricity_energy_kwh, ...]
        - If include_granular=True:
            [period, ahu_id, kWh_CCV, kWh_PC_CCV, kWh_HCV, kWh_DH_HCV,
             비용(원)_CCV, 비용(원)_PC_CCV, 비용(원)_HCV, 비용(원)_DH_HCV,
             전력_비용(원), 냉수_비용(원), 스팀_비용(원), 총합_비용(원)]

    Example:
        >>> df = fetch_batch_energy_consumption(
        ...     start_period="2021-06-01",
        ...     end_period="2021-07-01",
        ...     ahu_list=["AHU01", "AHU02", "AHU03"],
        ...     include_granular=True,
        ...     include_korean=True
        ... )
        >>> print(df[["period", "ahu_id", "kWh_CCV", "비용(원)_CCV"]])
                 period  ahu_id  kWh_CCV  비용(원)_CCV
        0  2021-06-01  AHU01  1234.5     150000
        1  2021-06-01  AHU02   987.3     120000
        ...
    """
    db = DatabaseManager()

    # Get all AHUs if not specified
    if ahu_list is None:
        metadata = fetch_all_ahu_metadata()
        ahu_list = list(metadata.keys())

    # Build IN clause for SQL
    ahu_list_str = ','.join(f"'{ahu}'" for ahu in ahu_list)

    # Build query with IN clause
    query = f"""
        SELECT
            date_trunc('day', timestamp) as period,
            ahu_id,
            SUM(ccv_cold_water_kwh) as ccv_cold_water_kwh,
            SUM(ac_ccv_cold_water_kwh) as ac_ccv_cold_water_kwh,
            SUM(pc_ccv_cold_water_kwh) as pc_ccv_cold_water_kwh,
            SUM(hcv_steam_kwh) as hcv_steam_kwh,
            SUM(ac_hcv_steam_kwh) as ac_hcv_steam_kwh,
            SUM(dh_hcv_steam_kwh) as dh_hcv_steam_kwh,
            SUM(sf_electricity_kwh) as sf_electricity_kwh,
            SUM(ccv_cold_water_cost_krw) as ccv_cold_water_cost_krw,
            SUM(ac_ccv_cold_water_cost_krw) as ac_ccv_cold_water_cost_krw,
            SUM(pc_ccv_cold_water_cost_krw) as pc_ccv_cold_water_cost_krw,
            SUM(hcv_steam_cost_krw) as hcv_steam_cost_krw,
            SUM(ac_hcv_steam_cost_krw) as ac_hcv_steam_cost_krw,
            SUM(dh_hcv_steam_cost_krw) as dh_hcv_steam_cost_krw,
            SUM(sf_cost_krw) as sf_cost_krw,
            SUM(electricity_cost_krw) as electricity_cost_krw,
            AVG(outdoor_temperature) as outdoor_temperature,
            AVG(outdoor_humidity) as outdoor_humidity
        FROM ahu_data.energy_readings
        WHERE timestamp >= %s AND timestamp < %s
          AND ahu_id IN ({ahu_list_str})
        GROUP BY date_trunc('day', timestamp), ahu_id
        ORDER BY period, ahu_id
    """

    try:
        result = db.fetch_all(query, (start_period, end_period))

        if not result:
            return pd.DataFrame()

        df = pd.DataFrame(result, columns=[
            'period', 'ahu_id', 'ccv_cold_water_kwh', 'ac_ccv_cold_water_kwh',
            'pc_ccv_cold_water_kwh', 'hcv_steam_kwh', 'ac_hcv_steam_kwh',
            'dh_hcv_steam_kwh', 'sf_electricity_kwh', 'ccv_cold_water_cost_krw',
            'ac_ccv_cold_water_cost_krw', 'pc_ccv_cold_water_cost_krw',
            'hcv_steam_cost_krw', 'ac_hcv_steam_cost_krw', 'dh_hcv_steam_cost_krw',
            'sf_cost_krw', 'electricity_cost_krw', 'outdoor_temperature',
            'outdoor_humidity'
        ])

        # Transform to granular format if requested
        if include_granular:
            df = _transform_batch_to_granular(df, include_korean=include_korean)

        return df

    except Exception as e:
        print(f"Error fetching batch energy data: {e}")
        raise


def _transform_batch_to_granular(df: pd.DataFrame, include_korean: bool = False):
    """Transform batch query result to granular format."""
    # Same transformation as individual query
    from ahu_query_lib.api.energy import _transform_to_granular_format
    return _transform_to_granular_format(df, include_korean=include_korean)
```

---

### 5. Add Timezone Handling 🔴 CRITICAL

**New File:** `ahu_query_lib/utils/timezone.py`

```python
"""
Timezone handling utilities for database queries.
"""
from typing import Literal
import pandas as pd
import pytz


def convert_timezone(
    df: pd.DataFrame,
    timestamp_column: str,
    timezone_as: Literal["aware", "naive", "utc"] = "naive",
    from_timezone: str = "Asia/Seoul"
) -> pd.DataFrame:
    """
    Convert timestamp column timezone.

    Args:
        df: Input DataFrame
        timestamp_column: Name of timestamp column
        timezone_as: Target timezone format
            - "aware": Keep timezone information (default DB behavior)
            - "naive": Remove timezone (for pandas compatibility)
            - "utc": Convert to UTC
        from_timezone: Source timezone (default: Asia/Seoul)

    Returns:
        DataFrame with converted timestamp column

    Example:
        >>> df["period"] = ["2021-06-01 00:00:00+09:00"]
        >>> df = convert_timezone(df, "period", timezone_as="naive")
        >>> df["period"]
        0   2021-06-01 00:00:00
    """
    df = df.copy()

    # Ensure timestamp column is datetime type
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    if timezone_as == "naive":
        # Remove timezone information
        if df[timestamp_column].dt.tz is not None:
            df[timestamp_column] = df[timestamp_column].dt.tz_localize(None)

    elif timezone_as == "utc":
        # Convert to UTC
        if df[timestamp_column].dt.tz is not None:
            df[timestamp_column] = df[timestamp_column].dt.tz_convert("UTC")
        else:
            # Assume it's in from_timezone and convert
            tz = pytz.timezone(from_timezone)
            df[timestamp_column] = df[timestamp_column].dt.tz_localize(tz).dt.tz_convert("UTC")

    # "aware" - do nothing, keep as-is

    return df
```

**Usage in `energy.py`:**
```python
from ahu_query_lib.utils.timezone import convert_timezone

def fetch_energy_consumption_cost(
    ...,
    timezone_as: Literal["aware", "naive", "utc"] = "naive"  # NEW PARAMETER
):
    # ... query and get DataFrame ...

    # Convert timezone before returning
    df = convert_timezone(df, "period", timezone_as=timezone_as)
    return df
```

---

### 6. Add Outdoor Air Data Function ⚠️ HIGH PRIORITY

**New File:** `ahu_query_lib/api/outdoor_air.py`

```python
"""
Outdoor air data functions.
"""
from typing import Literal
import pandas as pd
from ahu_query_lib.core.connection import DatabaseManager


def fetch_outdoor_air_data(
    start_date: str,
    end_date: str,
    aggregate: Literal["raw", "hourly", "daily"] = "raw",
    ahu_id: str = "AHU01",  # Use AHU01's sensor (same for all AHUs)
    timezone_as: Literal["aware", "naive", "utc"] = "naive",
    include_korean: bool = False
) -> pd.DataFrame:
    """
    Fetch outdoor temperature and humidity data.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        aggregate: Aggregation level
            - "raw": 6-minute resolution (default)
            - "hourly": Hourly averages
            - "daily": Daily averages
        ahu_id: AHU to use for outdoor sensor (default: "AHU01")
        timezone_as: Timezone format ("naive", "aware", "utc")
        include_korean: Use Korean column names

    Returns:
        DataFrame with columns:
        - If include_korean=False: [datetime, outdoor_temperature, outdoor_humidity]
        - If include_korean=True: [datetime, 외기온도, 외기습도]

    Example:
        >>> df = fetch_outdoor_air_data(
        ...     start_date="2021-06-01",
        ...     end_date="2021-06-02",
        ...     aggregate="daily",
        ...     include_korean=True
        ... )
        >>> print(df)
            datetime  외기온도  외기습도
        0 2021-06-01    24.5     65.2
        1 2021-06-02    26.1     62.8
    """
    db = DatabaseManager()

    # Build query based on aggregation level
    if aggregate == "raw":
        time_trunc = "timestamp"
    elif aggregate == "hourly":
        time_trunc = "date_trunc('hour', timestamp)"
    elif aggregate == "daily":
        time_trunc = "date_trunc('day', timestamp)"

    query = f"""
        SELECT
            {time_trunc} as timestamp,
            AVG(outdoor_temperature) as outdoor_temperature,
            AVG(outdoor_humidity) as outdoor_humidity
        FROM ahu_data.energy_readings
        WHERE timestamp >= %s AND timestamp < %s
          AND ahu_id = %s
          AND outdoor_temperature IS NOT NULL
        GROUP BY {time_trunc}
        ORDER BY timestamp
    """

    try:
        result = db.fetch_all(query, (start_date, end_date, ahu_id))

        if not result:
            return pd.DataFrame()

        column_names = ["outdoor_temperature", "outdoor_humidity"]
        if include_korean:
            column_names = ["외기온도", "외기습도"]

        df = pd.DataFrame(result, columns=["timestamp"] + column_names)

        # Convert timezone if needed
        from ahu_query_lib.utils.timezone import convert_timezone
        df = convert_timezone(df, "timestamp", timezone_as=timezone_as)

        return df

    except Exception as e:
        print(f"Error fetching outdoor air data: {e}")
        raise
```

---

### 7. Add Connection Pooling ⚠️ HIGH PRIORITY

**File:** `ahu_query_lib/db/connection.py`

**Current Code (INEFFICIENT):**
```python
import psycopg2

def get_connection():
    return psycopg2.connect(
        host='localhost',
        port=5433,
        user='postgres',
        password='admin',
        database='ahu_monitoring'
    )
```

**Required Fix:**
```python
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager

# Create connection pool (module-level singleton)
_connection_pool = None


def initialize_connection_pool(
    minconn: int = 2,
    maxconn: int = 10,
    **kwargs
):
    """
    Initialize connection pool (call once at startup).

    Args:
        minconn: Minimum number of connections in pool
        maxconn: Maximum number of connections in pool
        **kwargs: Connection parameters (host, port, user, password, database)
    """
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.SimpleConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            **kwargs
        )


@contextmanager
def get_connection():
    """
    Get connection from pool (context manager).

    Usage:
        with get_connection() as conn:
            # Use connection
            cursor = conn.cursor()
            cursor.execute(query)
        # Connection automatically returned to pool
    """
    if _connection_pool is None:
        # Fallback to direct connection if pool not initialized
        conn = psycopg2.connect(
            host='localhost',
            port=5433,
            user='postgres',
            password='admin',
            database='ahu_monitoring'
        )
        try:
            yield conn
        finally:
            conn.close()
    else:
        conn = _connection_pool.getconn()
        try:
            yield conn
        finally:
            _connection_pool.putconn(conn)


def close_all_connections():
    """Close all connections in pool (call on shutdown)."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None
```

**Update DatabaseManager to use pool:**
```python
class DatabaseManager:
    def __init__(self):
        # Use connection pool instead of direct connections
        from ahu_query_lib.db.connection import get_connection
        self.get_connection = get_connection

    def fetch_all(self, query, params=None):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params or ())
            result = cur.fetchall()
            cur.close()
            return result
```

---

### 8. Add Query Result Caching 🟢 NICE TO HAVE

**New File:** `ahu_query_lib/cache/query_cache.py`

```python
"""
Query result caching to avoid repeated expensive queries.
"""
from functools import lru_cache
from typing import Optional, Callable, Any
import hashlib
import pandas as pd


class QueryCache:
    """Cache for expensive database queries."""

    def __init__(self, maxsize: int = 128, ttl: int = 300):
        """
        Initialize cache.

        Args:
            maxsize: Maximum number of cached results
            ttl: Time-to-live in seconds (default: 5 minutes)
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache = {}

    def _make_key(self, query: str, params: tuple) -> str:
        """Create cache key from query and parameters."""
        key_str = f"{query}_{params}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, params: tuple) -> Optional[pd.DataFrame]:
        """Get cached result if available and not expired."""
        key = self._make_key(query, params)

        if key in self._cache:
            result, timestamp = self._cache[key]

            # Check if expired
            import time
            if time.time() - timestamp < self.ttl:
                return result
            else:
                # Expired, remove from cache
                del self._cache[key]

        return None

    def set(self, query: str, params: tuple, result: pd.DataFrame):
        """Cache query result."""
        key = self._make_key(query, params)

        # Evict oldest if cache is full
        if len(self._cache) >= self.maxsize:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        # Store result with timestamp
        import time
        self._cache[key] = (result, time.time())

    def clear(self):
        """Clear all cached results."""
        self._cache.clear()


# Global cache instance
_cache = QueryCache()


def cached_query(func: Callable) -> Callable:
    """
    Decorator to cache query results.

    Usage:
        @cached_query
        def fetch_energy_data(...):
            # Expensive query
            return df

        # First call: executes query
        df1 = fetch_energy_data(...)

        # Second call with same params: returns cached result
        df2 = fetch_energy_data(...)  # Much faster!
    """
    def wrapper(*args, **kwargs):
        # Create cache key from function arguments
        params_str = f"{args}_{kwargs}"
        key = hashlib.md5(params_str.encode()).hexdigest()

        # Check cache
        cached = _cache.get(key, ())
        if cached is not None:
            return cached

        # Execute query
        result = func(*args, **kwargs)

        # Cache result
        _cache.set(key, (), result)

        return result

    return wrapper
```

---

### 9. Add Async Support 🟢 NICE TO HAVE

**New File:** `ahu_query_lib/api/async_energy.py`

```python
"""
Async versions of query functions for non-blocking operations.
"""
import asyncpg
import pandas as pd
from typing import List, Optional, Literal


async def fetch_energy_consumption_cost_async(
    ahu_id: str,
    start_period: str,
    end_period: str,
    data_period: str = "daily",
    include_granular: bool = False,
    include_korean: bool = False
) -> pd.DataFrame:
    """
    Async version of fetch_energy_consumption_cost.

    Use this for non-blocking queries in async contexts.

    Args:
        Same as fetch_energy_consumption_cost

    Returns:
        Same as fetch_energy_consumption_cost

    Example:
        import asyncio

        async def main():
            df = await fetch_energy_consumption_cost_async(
                ahu_id="AHU01",
                start_period="2021-06-01",
                end_period="2021-07-01"
            )
            print(df)

        asyncio.run(main())
    """
    conn = await asyncpg.connect(
        'postgresql://postgres:admin@localhost:5433/ahu_monitoring'
    )

    try:
        query = """
            SELECT date_trunc('day', timestamp) as period,
                   ahu_id,
                   SUM(ccv_cold_water_kwh) as ccv_cold_water_kwh,
                   ...
            FROM ahu_data.energy_readings
            WHERE timestamp >= $1 AND timestamp < $2
              AND ahu_id = $3
            GROUP BY date_trunc('day', timestamp), ahu_id
        """

        rows = await conn.fetch(query, start_period, end_period, ahu_id)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            'period', 'ahu_id', 'ccv_cold_water_kwh', ...
        ])

        if include_granular:
            df = _transform_to_granular_format(df, include_korean=include_korean)

        return df

    finally:
        await conn.close()
```

---

## Database Setup Script

**New File:** `ahu-backend-server/scripts/setup_indexes.sql`

```sql
-- ============================================================================
-- Database Indexes for Performance
-- ============================================================================
-- Run this script ONCE to add indexes for fast queries
-- ============================================================================

-- Connect to database:
-- psql -h localhost -p 5433 -U postgres -d ahu_monitoring -f setup_indexes.sql

-- 1. Basic timestamp index (essential)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_energy_timestamp
    ON ahu_data.energy_readings(timestamp);

-- 2. Composite index for AHU + timestamp (for single AHU queries)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_energy_ahu_time
    ON ahu_data.energy_readings(ahu_id, timestamp);

-- 3. Covering index for daily queries (includes commonly aggregated columns)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_energy_daily_covering
    ON ahu_data.energy_readings(date_trunc('day', timestamp), ahu_id)
    INCLUDE (
        ccv_cold_water_kwh,
        hcv_steam_kwh,
        sf_electricity_kwh,
        ccv_cold_water_cost_krw,
        hcv_steam_cost_krw,
        sf_cost_krw
    );

-- 4. Partial index for recent data (speeds up most queries)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_energy_recent
    ON ahu_data.energy_readings(timestamp DESC, ahu_id)
    WHERE timestamp >= CURRENT_DATE - INTERVAL '2 years';

-- 5. Index for outdoor air queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_energy_outdoor
    ON ahu_data.energy_readings(timestamp)
    WHERE outdoor_temperature IS NOT NULL;

-- ============================================================================
-- Verify Indexes
-- ============================================================================

-- Check created indexes:
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'ahu_data'
  AND tablename = 'energy_readings'
ORDER BY indexname;

-- Analyze table to update statistics:
ANALYZE ahu_data.energy_readings;

-- ============================================================================
-- Test Query Performance
-- ============================================================================

EXPLAIN ANALYZE
SELECT
    date_trunc('day', timestamp) as period,
    ahu_id,
    SUM(ccv_cold_water_kwh) as ccv_cold_water_kwh
FROM ahu_data.energy_readings
WHERE timestamp >= '2021-06-01' AND timestamp < '2022-05-01'
  AND ahu_id = 'AHU01'
GROUP BY date_trunc('day', timestamp), ahu_id;

-- Expected result:
-- Should show "Index Scan" or "Bitmap Index Scan" instead of "Seq Scan"
-- Execution time should be < 100ms
```

---

## Testing Checklist

After implementing all changes, verify:

- [ ] `fetch_energy_consumption_cost()` returns data (not empty DataFrame)
- [ ] Column names match database schema (`outdoor_temperature`, not `avg_outdoor_temp`)
- [ ] Batch query `fetch_batch_energy_consumption()` works for multiple AHUs
- [ ] Granular breakdown returns CCV/PC_CCV/HCV/DH_HCV separately
- [ ] Korean column names work (`비용(원)_CCV`, etc.)
- [ ] Timezone conversion produces naive timestamps
- [ ] Outdoor air function returns `외기온도`, `외기습도` columns
- [ ] Queries complete in < 1 second (with indexes)
- [ ] Connection pooling works (multiple connections reuse)
- [ ] Cache returns same result for repeated queries

---

## Migration Steps

1. **Backup existing code**
   ```bash
   cp -r ahu-query-lib ahu-query-lib.backup
   ```

2. **Create new files**
   - `ahu_query_lib/api/batch.py`
   - `ahu_query_lib/api/outdoor_air.py`
   - `ahu_query_lib/cache/query_cache.py`
   - `ahu_query_lib/utils/timezone.py`

3. **Modify existing files**
   - `ahu_query_lib/queries/energy_queries.py` - Fix column names, change source table
   - `ahu_query_lib/db/connection.py` - Add connection pooling
   - `ahu_query_lib/api/energy.py` - Add new parameters

4. **Run database setup**
   ```bash
   psql -h localhost -p 5433 -U postgres -d ahu_monitoring -f scripts/setup_indexes.sql
   ```

5. **Test thoroughly**
   ```python
   python -m pytest tests/test_batch_queries.py
   python -m pytest tests/test_granular_costs.py
   python -m pytest tests/test_timezone.py
   ```

6. **Deploy**
   ```bash
   git add .
   git commit -m "Add viz integration features"
   git tag v1.1.0
   git push origin main
   ```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-07
**Status:** Ready for Implementation
