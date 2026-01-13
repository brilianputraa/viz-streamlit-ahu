#!/usr/bin/env python3
"""
Parquet preprocessing script for DB vs Parquet comparison.
Sets AHU_HISTORY_DIR to point to BMS-4_history RAW data.
Includes error handling and timing for benchmarking.
"""
import os
import sys
import time

# Set environment variable to point to correct RAW data location
# Use RAW folder - it has point,Date,Value format (some files may be skipped)
os.environ["AHU_HISTORY_DIR"] = r"C:\Users\User\Documents\BMS-4_history\RAW"

# Also set result base to a consistent location
os.environ["AHU_RESULT_BASE"] = r"C:\Users\User\Documents\viz-streamlit-ahu\results"

print(f"AHU_HISTORY_DIR set to: {os.environ['AHU_HISTORY_DIR']}")
print(f"AHU_RESULT_BASE set to: {os.environ.get('AHU_RESULT_BASE', 'default')}")

# Import loader after setting environment variables
import loader

def main():
    print("=" * 60)
    print("Starting Parquet Preprocessing for DB Comparison")
    print("=" * 60)

    # Start timing
    start_time = time.time()
    files_processed = 0
    files_skipped = 0
    errors = []

    # Use mutable containers for counters to avoid nonlocal issues
    counters = {"skipped": 0}

    # Monkey patch the loader to add error handling
    original_parse_ahu_csv = loader.parse_ahu_csv

    def parse_ahu_csv_with_error_handling(path):
        try:
            return original_parse_ahu_csv(path)
        except ValueError as e:
            fname = os.path.basename(path)
            errors.append(f"{fname}: {str(e)}")
            counters["skipped"] += 1
            print(f"  [SKIP] {fname}: {str(e)[:80]}...")
            return pd.DataFrame()  # Return empty DataFrame to skip this file
        except Exception as e:
            fname = os.path.basename(path)
            errors.append(f"{fname}: {str(e)}")
            counters["skipped"] += 1
            print(f"  [ERROR] {fname}: {str(e)[:80]}...")
            return pd.DataFrame()

    # Monkey patch
    loader.parse_ahu_csv = parse_ahu_csv_with_error_handling

    # Import pandas for empty DataFrame
    import pandas as pd

    # Run the scan and update (preprocessing)
    print("\n[1/2] Scanning RAW CSV files and generating parquet...")
    try:
        final_df, oa_daily_df, oa_results_df = loader.scan_and_update(
            progress_callback=lambda i, n, fname: [setattr(__builtins__, '_files_processed', i), print(f"  [{i}/{n}] Processing: {fname}")][1]
        )
        files_processed = getattr(__builtins__, '_files_processed', 0)
    except Exception as e:
        print(f"\n[ERROR] Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    # Calculate timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Get final skip count from counters
    files_skipped = counters["skipped"]

    print(f"\n[2/2] Preprocessing complete!")
    print(f"\n" + "=" * 60)
    print("TIMING STATISTICS")
    print("=" * 60)
    print(f"Total elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Files processed: {files_processed}")
    print(f"Files skipped: {files_skipped}")

    if files_processed > 0:
        print(f"Average time per file: {elapsed_time/files_processed:.2f} seconds")

    if errors:
        print(f"\nErrors/Warnings ({len(errors)}):")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    print(f"\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  - Final results shape: {final_df.shape if not final_df.empty else 'Empty'}")
    print(f"  - OA daily shape: {oa_daily_df.shape if not oa_daily_df.empty else 'Empty'}")
    print(f"  - OA results shape: {oa_results_df.shape if not oa_results_df.empty else 'Empty'}")

    if not final_df.empty:
        print(f"\nFinal results columns ({len(final_df.columns)}):")
        for col in final_df.columns:
            print(f"  - {col}")

        print(f"\nDate range: {final_df['datetime'].min()} to {final_df['datetime'].max()}")
        print(f"AHUs in data: {sorted(final_df['공조기'].unique())}")

    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print("=" * 60)

    return final_df, oa_daily_df, oa_results_df

if __name__ == "__main__":
    main()
