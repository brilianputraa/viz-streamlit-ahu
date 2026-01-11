import io
import pandas as pd


def test_load_final_results_from_dir_empty(tmp_path):
    from app2_loader import load_final_results_from_dir

    df = load_final_results_from_dir(str(tmp_path))
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_final_results_from_dir_reads_parquet(tmp_path):
    from app2_loader import load_final_results_from_dir

    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pd.DataFrame({"a": [3], "b": ["z"]})

    buf1 = io.BytesIO()
    df1.to_parquet(buf1, index=False)
    (tmp_path / "final_analysis_AHU01.parquet").write_bytes(buf1.getvalue())

    buf2 = io.BytesIO()
    df2.to_parquet(buf2, index=False)
    (tmp_path / "final_analysis_AHU02.parquet").write_bytes(buf2.getvalue())

    result = load_final_results_from_dir(str(tmp_path))

    assert list(result.columns) == ["a", "b"]
    assert len(result) == 3


def test_load_parquet_data_update_path():
    from app2_loader import load_parquet_data

    sentinel_final = pd.DataFrame({"x": [1]})
    sentinel_daily = pd.DataFrame({"y": [2]})
    sentinel_all = pd.DataFrame({"z": [3]})

    calls = {"update": 0, "final": 0, "daily": 0, "all": 0}

    def update_fn():
        calls["update"] += 1
        return sentinel_final, sentinel_daily, sentinel_all

    def final_fn():
        calls["final"] += 1
        return pd.DataFrame()

    def daily_fn():
        calls["daily"] += 1
        return pd.DataFrame()

    def all_fn():
        calls["all"] += 1
        return pd.DataFrame()

    result_final, result_daily, result_all, did_update = load_parquet_data(
        should_update=True,
        update_fn=update_fn,
        final_fn=final_fn,
        oa_daily_fn=daily_fn,
        oa_all_fn=all_fn,
    )

    assert did_update is True
    assert calls == {"update": 1, "final": 0, "daily": 0, "all": 0}
    assert result_final.equals(sentinel_final)
    assert result_daily.equals(sentinel_daily)
    assert result_all.equals(sentinel_all)


def test_load_parquet_data_cached_path():
    from app2_loader import load_parquet_data

    sentinel_final = pd.DataFrame({"x": [1]})
    sentinel_daily = pd.DataFrame({"y": [2]})
    sentinel_all = pd.DataFrame({"z": [3]})

    calls = {"update": 0, "final": 0, "daily": 0, "all": 0}

    def update_fn():
        calls["update"] += 1
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def final_fn():
        calls["final"] += 1
        return sentinel_final

    def daily_fn():
        calls["daily"] += 1
        return sentinel_daily

    def all_fn():
        calls["all"] += 1
        return sentinel_all

    result_final, result_daily, result_all, did_update = load_parquet_data(
        should_update=False,
        update_fn=update_fn,
        final_fn=final_fn,
        oa_daily_fn=daily_fn,
        oa_all_fn=all_fn,
    )

    assert did_update is False
    assert calls == {"update": 0, "final": 1, "daily": 1, "all": 1}
    assert result_final.equals(sentinel_final)
    assert result_daily.equals(sentinel_daily)
    assert result_all.equals(sentinel_all)
