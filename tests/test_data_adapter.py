import pytest
from data_adapter import DataAccessMode, load_final_results
from data_adapter import ensure_ahu_query_lib
from pathlib import Path

def test_data_access_mode_enum():
    assert DataAccessMode.PARQUET.value == "parquet"
    assert DataAccessMode.DATABASE.value == "database"

def test_load_final_results_parquet_mode():
    # This should use existing parquet loader
    df = load_final_results(mode=DataAccessMode.PARQUET)
    # Verify it returns a DataFrame (may be empty if no data)
    assert hasattr(df, 'columns')

def test_ensure_ahu_query_lib_available_without_import_side_effects(monkeypatch):
    # Simulate launching Streamlit from inside viz-streamlit-ahu/
    viz_dir = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(viz_dir)

    # Availability check should not require importing (which may attempt DB connection).
    spec = ensure_ahu_query_lib(import_module=False)
    assert spec is not None
