import glob
import os
from typing import Callable, Tuple

import pandas as pd


ParquetUpdateFn = Callable[[], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
ParquetLoadFn = Callable[[], pd.DataFrame]


def load_final_results_from_dir(final_dir: str) -> pd.DataFrame:
    """Load final_analysis_*.parquet files from a directory if present."""
    if not final_dir or not os.path.isdir(final_dir):
        return pd.DataFrame()

    files = glob.glob(os.path.join(final_dir, "final_analysis_*.parquet"))
    if not files:
        return pd.DataFrame()

    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def load_parquet_data(
    should_update: bool,
    update_fn: ParquetUpdateFn,
    final_fn: ParquetLoadFn,
    oa_daily_fn: ParquetLoadFn,
    oa_all_fn: ParquetLoadFn,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Load parquet data, optionally refreshing via update_fn."""
    if should_update:
        df_final_all, df_oa_daily, df_oa_all = update_fn()
        return df_final_all, df_oa_daily, df_oa_all, True

    df_final_all = final_fn()
    df_oa_daily = oa_daily_fn()
    df_oa_all = oa_all_fn()
    return df_final_all, df_oa_daily, df_oa_all, False
