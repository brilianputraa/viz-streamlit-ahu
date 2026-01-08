import pytest
from data_adapter import DataAccessMode, load_final_results

def test_data_access_mode_enum():
    assert DataAccessMode.PARQUET.value == "parquet"
    assert DataAccessMode.DATABASE.value == "database"

def test_load_final_results_parquet_mode():
    # This should use existing parquet loader
    df = load_final_results(mode=DataAccessMode.PARQUET)
    # Verify it returns a DataFrame (may be empty if no data)
    assert hasattr(df, 'columns')
