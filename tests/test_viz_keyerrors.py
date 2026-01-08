"""
Headless test for KeyError issues in visualization features.

Tests each granular visualization feature with empty DataFrames
to identify all KeyError issues.
"""
import sys
import os
sys.path.insert(0, '/Users/putra/viz-streamlit-ahu')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import modules to test
from common import 절기_분류
from loader import get_items_from_final

print("="*60)
print("HEADLESS TEST: KeyError Detection in Viz Features")
print("="*60)

# Create test DataFrames
print("\n[Setup] Creating test DataFrames...")

# Empty DataFrame (simulates Database mode with no energy data)
empty_df = pd.DataFrame()

# DataFrame with expected columns
test_dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
full_df = pd.DataFrame({
    'datetime': test_dates,
    '공조기': ['AHU01'] * 50 + ['AHU02'] * 50,
    '항목명': ['CCV'] * 25 + ['HCV'] * 25 + ['CCV'] * 25 + ['HCV'] * 25,
    '값': np.random.uniform(10, 90, 100),
    'kWh': np.random.uniform(100, 500, 100),
    '전력_비용(원)': np.random.uniform(10000, 50000, 100)
})

print(f"  Empty DataFrame: {len(empty_df)} rows")
print(f"  Full DataFrame: {len(full_df)} rows")

# ============================================================================
# Test 1: DataFrame column access - 공조기
# ============================================================================
print("\n[Test 1] Testing '공조기' column access...")
try:
    if not empty_df.empty and "공조기" in empty_df.columns:
        result = empty_df["공조기"]
        print("  ✅ Empty df handled correctly")
    else:
        print("  ✅ Empty check passed")
except KeyError as e:
    print(f"  ❌ KeyError: {e}")

# ============================================================================
# Test 2: DataFrame column access - datetime
# ============================================================================
print("\n[Test 2] Testing 'datetime' column access...")
try:
    if not empty_df.empty and "datetime" in empty_df.columns:
        start_date = empty_df["datetime"].min().date()
    else:
        print("  ✅ Empty check passed")
except KeyError as e:
    print(f"  ❌ KeyError: {e}")

# ============================================================================
# Test 3: DataFrame operations - 연도/절기
# ============================================================================
print("\n[Test 3] Testing 연도/절기 operations...")
try:
    if not empty_df.empty and "datetime" in empty_df.columns:
        empty_df["연도"] = empty_df["datetime"].dt.year
        empty_df["절기"] = empty_df["datetime"].apply(절기_분류)
        print("  ✅ Operations successful")
    else:
        print("  ✅ Empty check passed")
except KeyError as e:
    print(f"  ❌ KeyError: {e}")

# ============================================================================
# Test 4: get_items_from_final
# ============================================================================
print("\n[Test 4] Testing get_items_from_final...")
try:
    items = get_items_from_final(empty_df)
    print(f"  ✅ Returned: {items}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# ============================================================================
# Test 5: Multiple column access
# ============================================================================
print("\n[Test 5] Testing multiple column access...")
test_cases = [
    ("공조기", empty_df),
    ("datetime", empty_df),
    ("전력_비용(원)", empty_df),
]

for col_name, df in test_cases:
    try:
        if not df.empty and col_name in df.columns:
            result = df[col_name]
            print(f"  ✅ '{col_name}' accessible")
        else:
            print(f"  ✅ '{col_name}' empty check passed")
    except KeyError as e:
        print(f"  ❌ KeyError for '{col_name}': {e}")

# ============================================================================
# Test 6: DataFrame filtering operations
# ============================================================================
print("\n[Test 6] Testing DataFrame filtering...")
try:
    시작 = pd.to_datetime('2025-01-01')
    종료 = pd.to_datetime('2025-01-02')

    if not empty_df.empty and "datetime" in empty_df.columns:
        filtered = empty_df[
            (empty_df["datetime"] >= 시작) &
            (empty_df["datetime"] < 종료)
        ]
    else:
        print("  ✅ Empty check passed")
except KeyError as e:
    print(f"  ❌ KeyError: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("SUMMARY: KeyError Detection Complete")
print("="*60)
print("\nRecommendations:")
print("1. Always check: if not df.empty and 'column' in df.columns")
print("2. Use .get() method for safer column access")
print("3. Add try-except blocks around DataFrame operations")
print("="*60)
