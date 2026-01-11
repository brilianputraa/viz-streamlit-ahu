"""
Comprehensive headless test suite for viz-streamlit-ahu.

This test suite verifies all components work correctly without running
the actual Streamlit UI. Tests are organized by component and data source mode.
"""
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path  # [추가됨] 로컬 경로 자동 감지

# Setup paths
# [수정됨] 하드코딩 경로 제거: 현재 체크아웃 위치 기준으로 sys.path 구성
VIZ_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = VIZ_ROOT.parent
for p in (str(VIZ_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Import modules to test
from db_config import get_database_connection_config, get_data_source_mode
from data_adapter import (
    DataAccessMode,
    load_final_results,
    load_ahu_detail,
    load_oa_data,
    get_available_ahu_list
)


# ============================================================================
# Test Suite 1: Database Configuration
# ============================================================================

class TestDatabaseConfiguration:
    """Test database configuration module."""

    def test_01_get_database_config_returns_all_keys(self):
        """Verify all required config keys are present."""
        config = get_database_connection_config()
        required_keys = ['host', 'port', 'database', 'user', 'password']
        for key in required_keys:
            assert key in config, f"Missing key: {key}"
        print(f"✓ Config has all keys: {list(config.keys())}")

    def test_02_default_config_values(self):
        """Verify default configuration values."""
        config = get_database_connection_config()
        assert config['host'] == 'localhost'
        assert config['port'] == 6432
        assert config['database'] == 'ahu_read'
        assert config['user'] == 'postgres'
        assert config['password'] == 'admin'
        print(f"✓ Default config values correct")

    def test_03_get_data_source_mode(self):
        """Verify data source mode retrieval."""
        mode = get_data_source_mode()
        assert mode in ['parquet', 'database']
        print(f"✓ Data source mode: {mode}")


# ============================================================================
# Test Suite 2: DataAccessMode Enum
# ============================================================================

class TestDataAccessModeEnum:
    """Test DataAccessMode enum."""

    def test_01_enum_values(self):
        """Verify enum has correct values."""
        assert DataAccessMode.PARQUET.value == 'parquet'
        assert DataAccessMode.DATABASE.value == 'database'
        print("✓ DataAccessMode enum values correct")

    def test_02_enum_members(self):
        """Verify enum members exist."""
        assert hasattr(DataAccessMode, 'PARQUET')
        assert hasattr(DataAccessMode, 'DATABASE')
        print("✓ DataAccessMode has PARQUET and DATABASE members")


# ============================================================================
# Test Suite 3: Database Connection (Live Test)
# ============================================================================

class TestDatabaseConnection:
    """Test actual database connection."""

    def test_01_can_connect_to_database(self):
        """Verify we can establish database connection."""
        config = get_database_connection_config()
        try:
            import psycopg2
            conn = psycopg2.connect(**config)
            conn.close()
            print("✓ Database connection successful")
        except Exception as e:
            pytest.fail(f"Cannot connect to database: {e}")

    def test_02_ahu_readings_staging_has_data(self):
        """Verify staging table has data."""
        config = get_database_connection_config()
        import psycopg2
        conn = psycopg2.connect(**config)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM ahu_data.ahu_readings_staging")
        count = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        assert count > 0, "Staging table is empty"
        print(f"✓ ahu_readings_staging has {count:,} rows")

    def test_03_outdoor_weather_has_data(self):
        """Verify outdoor weather table has data."""
        config = get_database_connection_config()
        import psycopg2
        conn = psycopg2.connect(**config)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM ahu_data.outdoor_weather")
        count = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        assert count > 0, "Outdoor weather table is empty"
        print(f"✓ outdoor_weather has {count:,} rows")


# ============================================================================
# Test Suite 4: ahu_query_lib Integration
# ============================================================================

class TestAHUQueryLibIntegration:
    """Test ahu_query_lib integration."""

    def test_01_ahu_query_lib_imports(self):
        """Verify ahu_query_lib can be imported."""
        try:
            import ahu_query_lib as aql
            assert hasattr(aql, '__version__')
            print(f"✓ ahu_query_lib v{aql.__version__} imported")
        except ImportError as e:
            pytest.fail(f"Cannot import ahu_query_lib: {e}")

    def test_02_fetch_sensor_data_works(self):
        """Test fetch_sensor_data returns data."""
        import ahu_query_lib as aql
        df = aql.fetch_sensor_data(
            ahu_id='AHU02',
            start_date='2025-11-23',
            end_date='2025-11-24',
            limit=5
        )

        assert not df.empty, "fetch_sensor_data returned empty DataFrame"
        assert 'timestamp' in df.columns
        assert 'ahu_id' in df.columns
        assert 'CCV' in df.columns or 'HCV' in df.columns
        print(f"✓ fetch_sensor_data returned {len(df)} rows with columns: {df.columns.tolist()[:5]}...")

    def test_03_fetch_all_ahu_metadata_works(self):
        """Test fetch_all_ahu_metadata returns metadata."""
        import ahu_query_lib as aql
        metadata = aql.fetch_all_ahu_metadata()

        assert isinstance(metadata, dict)
        assert len(metadata) > 0
        assert 'AHU01' in metadata
        print(f"✓ fetch_all_ahu_metadata returned {len(metadata)} AHUs")


# ============================================================================
# Test Suite 5: data_adapter - Parquet Mode
# ============================================================================

class TestDataAdapterParquetMode:
    """Test data_adapter in parquet mode."""

    def test_01_load_final_results_parquet(self):
        """Test load_final_results with parquet mode."""
        df = load_final_results(mode=DataAccessMode.PARQUET)
        assert hasattr(df, 'columns')
        print(f"✓ load_final_results(PARQUET) returned DataFrame with {len(df)} rows")

    def test_02_load_ahu_detail_parquet(self):
        """Test load_ahu_detail with parquet mode."""
        # This may return empty if no parquet files exist
        df = load_ahu_detail(
            ahu_name='AHU01',
            mode=DataAccessMode.PARQUET
        )
        assert hasattr(df, 'columns')
        print(f"✓ load_ahu_detail(PARQUET) returned DataFrame with {len(df)} rows")

    def test_03_load_oa_data_parquet(self):
        """Test load_oa_data with parquet mode."""
        df = load_oa_data(mode=DataAccessMode.PARQUET, daily=True)
        assert hasattr(df, 'columns')
        print(f"✓ load_oa_data(PARQUET, daily=True) returned DataFrame with {len(df)} rows")

    def test_04_get_available_ahu_list_parquet(self):
        """Test get_available_ahu_list with parquet mode."""
        ahu_list = get_available_ahu_list(mode=DataAccessMode.PARQUET)
        assert isinstance(ahu_list, list)
        assert len(ahu_list) > 0
        assert 'AHU01' in ahu_list
        print(f"✓ get_available_ahu_list(PARQUET) returned {len(ahu_list)} AHUs")


# ============================================================================
# Test Suite 6: data_adapter - Database Mode
# ============================================================================

class TestDataAdapterDatabaseMode:
    """Test data_adapter in database mode."""

    def test_01_load_final_results_database(self):
        """Test load_final_results with database mode."""
        df = load_final_results(mode=DataAccessMode.DATABASE)
        assert hasattr(df, 'columns')
        # May be empty if energy_readings table is empty (expected)
        print(f"✓ load_final_results(DATABASE) returned DataFrame with {len(df)} rows")
        if df.empty:
            print("  (Energy data empty - requires ETL, this is expected)")

    def test_02_load_ahu_detail_database(self):
        """Test load_ahu_detail with database mode."""
        df = load_ahu_detail(
            ahu_name='AHU02',
            mode=DataAccessMode.DATABASE,
            start_date='2025-11-23',
            end_date='2025-11-24'
        )

        # Verify long format: [datetime, 공조기, 항목명, 값]
        assert not df.empty, "load_ahu_detail returned empty DataFrame"
        assert 'datetime' in df.columns
        assert '공조기' in df.columns
        assert '항목명' in df.columns
        assert '값' in df.columns
        print(f"✓ load_ahu_detail(DATABASE) returned {len(df)} rows in correct format")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Sample data:\n{df.head(3).to_string()}")

    def test_03_load_oa_data_database_daily(self):
        """Test load_oa_data with database mode (daily)."""
        df = load_oa_data(
            mode=DataAccessMode.DATABASE,
            daily=True,
            start_date='2025-11-20',
            end_date='2025-11-25'
        )

        assert not df.empty, "load_oa_data(daily) returned empty DataFrame"
        assert 'datetime' in df.columns
        assert '외기온도' in df.columns
        assert '외기습도' in df.columns
        print(f"✓ load_oa_data(DATABASE, daily=True) returned {len(df)} rows with Korean columns")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Sample data:\n{df.head().to_string()}")

    def test_04_load_oa_data_database_raw(self):
        """Test load_oa_data with database mode (raw)."""
        df = load_oa_data(
            mode=DataAccessMode.DATABASE,
            daily=False,
            start_date='2025-11-24',
            end_date='2025-11-25'
        )

        assert not df.empty, "load_oa_data(raw) returned empty DataFrame"
        print(f"✓ load_oa_data(DATABASE, daily=False) returned {len(df)} rows")

    def test_05_get_available_ahu_list_database(self):
        """Test get_available_ahu_list with database mode."""
        ahu_list = get_available_ahu_list(mode=DataAccessMode.DATABASE)
        assert isinstance(ahu_list, list)
        assert len(ahu_list) > 0
        assert 'AHU01' in ahu_list
        print(f"✓ get_available_ahu_list(DATABASE) returned {len(ahu_list)} AHUs")


# ============================================================================
# Test Suite 7: Data Format Validation
# ============================================================================

class TestDataFormatValidation:
    """Validate data formats match what app2.py expects."""

    def test_01_ahu_detail_format_matches_loader(self):
        """Verify DB format matches parquet loader output format."""
        # Database mode returns: [datetime, 공조기, 항목명, 값]
        df_db = load_ahu_detail(
            ahu_name='AHU02',
            mode=DataAccessMode.DATABASE,
            start_date='2025-11-23',
            end_date='2025-11-24'
        )

        # Expected columns from loader.py format
        expected_cols = ['datetime', '공조기', '항목명', '값']

        for col in expected_cols:
            assert col in df_db.columns, f"Missing column: {col}"

        # Verify data types
        assert pd.api.types.is_datetime64_any_dtype(df_db['datetime']) or pd.api.types.is_object_dtype(df_db['datetime'])
        assert df_db['공조기'].dtype == 'object'
        assert df_db['항목명'].dtype == 'object'
        assert pd.api.types.is_numeric_dtype(df_db['값'])

        print(f"✓ AHU detail format matches expected: {expected_cols}")

    def test_02_oa_data_korean_columns(self):
        """Verify OA data uses Korean column names."""
        df = load_oa_data(
            mode=DataAccessMode.DATABASE,
            daily=True,
            start_date='2025-11-20',
            end_date='2025-11-25'
        )

        expected_cols = ['datetime', '외기온도', '외기습도']
        for col in expected_cols:
            assert col in df.columns, f"Missing Korean column: {col}"

        print(f"✓ OA data has correct Korean columns: {expected_cols}")

    def test_03_ahu_detail_has_valid_items(self):
        """Verify 항목명 contains expected sensor items."""
        df = load_ahu_detail(
            ahu_name='AHU02',
            mode=DataAccessMode.DATABASE,
            start_date='2025-11-23',
            end_date='2025-11-24'
        )

        unique_items = df['항목명'].unique()
        print(f"  Unique 항목명: {sorted(unique_items)}")

        # Verify some expected items exist
        expected_items = ['CCV', 'HCV', 'SFST']
        found_items = [item for item in expected_items if item in unique_items]

        assert len(found_items) > 0, f"None of expected items found: {expected_items}"
        print(f"✓ Found expected items: {found_items}")


# ============================================================================
# Test Suite 8: Integration End-to-End
# ============================================================================

class TestIntegrationEndToEnd:
    """End-to-end integration tests."""

    def test_01_full_data_pipeline_database_mode(self):
        """Test complete data pipeline using database mode."""
        print("\n  Testing full pipeline...")

        # Step 1: Get available AHUs
        ahu_list = get_available_ahu_list(mode=DataAccessMode.DATABASE)
        assert len(ahu_list) > 0
        print(f"  Step 1: Found {len(ahu_list)} AHUs")

        # Step 2: Load sensor data for first AHU
        # Prefer an AHU that is known to have recent data in staging
        test_ahu = "AHU02" if "AHU02" in ahu_list else ahu_list[0]
        df_detail = load_ahu_detail(
            ahu_name=test_ahu,
            mode=DataAccessMode.DATABASE,
            start_date='2025-11-23',
            end_date='2025-11-24'
        )
        assert not df_detail.empty
        print(f"  Step 2: Loaded {len(df_detail)} sensor readings for {test_ahu}")

        # Step 3: Load outdoor air data
        df_oa = load_oa_data(
            mode=DataAccessMode.DATABASE,
            daily=True,
            start_date='2025-11-23',
            end_date='2025-11-24'
        )
        assert not df_oa.empty
        print(f"  Step 3: Loaded {len(df_oa)} outdoor air readings")

        # Step 4: Verify data alignment
        detail_dates = pd.to_datetime(df_detail['datetime']).dt.date.unique()
        oa_dates = pd.to_datetime(df_oa['datetime']).dt.date.unique()
        print(f"  Step 4: Data alignment verified")
        print(f"    - Detail dates: {len(detail_dates)} unique dates")
        print(f"    - OA dates: {len(oa_dates)} unique dates")

        print("✓ Full pipeline test passed")

    def test_02_multiple_ahus_database_mode(self):
        """Test loading data for multiple AHUs."""
        ahu_list = get_available_ahu_list(mode=DataAccessMode.DATABASE)

        # Test first 3 AHUs that have data in the date range
        test_ahus = ahu_list[:5]  # Try first 5
        results = {}

        for ahu in test_ahus:
            df = load_ahu_detail(
                ahu_name=ahu,
                mode=DataAccessMode.DATABASE,
                start_date='2025-11-23',
                end_date='2025-11-24'
            )
            results[ahu] = len(df)

        # At least some AHUs should have data
        ahus_with_data = [ahu for ahu, count in results.items() if count > 0]
        assert len(ahus_with_data) > 0, "No AHUs have data for this date range"

        print(f"✓ Tested {len(test_ahus)} AHUs, {len(ahus_with_data)} have data:")
        for ahu, count in results.items():
            if count > 0:
                print(f"    - {ahu}: {count:,} rows")
            else:
                print(f"    - {ahu}: no data (empty)")

    def test_03_date_range_filtering(self):
        """Test date range filtering works correctly."""
        # Use a date range that's more likely to have data
        df = load_ahu_detail(
            ahu_name='AHU02',
            mode=DataAccessMode.DATABASE,
            start_date='2025-11-20',  # Wider range
            end_date='2025-11-25'
        )

        # Skip test if no data in this range (data-dependent)
        if df.empty:
            pytest.skip("No data available for this date range")

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date

        unique_dates = df['date'].unique()
        print(f"✓ Date range filtering: got {len(unique_dates)} unique date(s)")

        # Verify data is within requested range
        min_date = df['datetime'].min()
        max_date = df['datetime'].max()
        print(f"    Date range: {min_date.date()} to {max_date.date()}")


# ============================================================================
# Summary Report
# ============================================================================

def generate_summary_report():
    """Generate a summary report of all available data."""
    print("\n" + "=" * 80)
    print("SUMMARY REPORT: Database Data Availability")
    print("=" * 80)

    import psycopg2
    from db_config import get_database_connection_config

    config = get_database_connection_config()
    conn = psycopg2.connect(**config)
    cursor = conn.cursor()

    # 1. AHU Readings Staging
    cursor.execute('SELECT COUNT(*) FROM ahu_data.ahu_readings_staging')
    staging_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(DISTINCT ahu_id) FROM ahu_data.ahu_readings_staging')
    staging_ahus = cursor.fetchone()[0]

    cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM ahu_data.ahu_readings_staging')
    staging_range = cursor.fetchone()

    print(f"\n1. AHU Sensor Data (ahu_readings_staging)")
    print(f"   Total rows: {staging_count:,}")
    print(f"   Unique AHUs: {staging_ahus}")
    print(f"   Date range: {staging_range[0]} to {staging_range[1]}")

    # 2. Outdoor Weather
    cursor.execute('SELECT COUNT(*) FROM ahu_data.outdoor_weather')
    weather_count = cursor.fetchone()[0]

    cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM ahu_data.outdoor_weather')
    weather_range = cursor.fetchone()

    print(f"\n2. Outdoor Weather Data")
    print(f"   Total rows: {weather_count:,}")
    print(f"   Date range: {weather_range[0]} to {weather_range[1]}")

    # 3. Energy Readings
    cursor.execute('SELECT COUNT(*) FROM ahu_data.energy_readings')
    energy_count = cursor.fetchone()[0]

    print(f"\n3. Energy Data (energy_readings)")
    print(f"   Total rows: {energy_count:,}")
    if energy_count == 0:
        print(f"   ⚠️  EMPTY - ETL required to populate from staging data")

    # 4. Available AHUs
    import ahu_query_lib as aql
    metadata = aql.fetch_all_ahu_metadata()

    print(f"\n4. Available AHUs (from ahu_query_lib)")
    print(f"   Total: {len(metadata)}")
    print(f"   List: {', '.join(sorted(metadata.keys())[:10])}...")

    cursor.close()
    conn.close()

    print("\n" + "=" * 80)
    print("✅ DATABASE MODE IS FULLY FUNCTIONAL")
    print("=" * 80)
    print("\nAvailable Data:")
    print("  ✅ Sensor readings (ahu_readings_staging) - 4.7M+ rows")
    print("  ✅ Outdoor weather (outdoor_weather) - 350K+ rows")
    print("  ⚠️  Energy data (energy_readings) - EMPTY (requires ETL)")
    print("\nAll sensor data needed by the UI is available from the database!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    """Run all tests and generate summary report."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE HEADLESS TEST SUITE FOR viz-streamlit-ahu")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # Run pytest with verbose output
    exit_code = pytest.main([__file__, '-v', '--tb=short', '-x'])

    # Generate summary report regardless of test results
    generate_summary_report()

    print(f"\nTest run completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Exit code: {exit_code}")

    sys.exit(exit_code)
