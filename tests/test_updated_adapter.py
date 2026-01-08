"""
Test the updated data_adapter with database mode.
"""
import sys
sys.path.insert(0, '/Users/putra/viz-streamlit-ahu')

from data_adapter import DataAccessMode, load_ahu_detail, load_oa_data, get_available_ahu_list
import warnings
warnings.filterwarnings('ignore')

print('=' * 60)
print('Test 1: load_ahu_detail with DATABASE mode')
print('=' * 60)

try:
    df = load_ahu_detail(
        ahu_name='AHU01',
        mode=DataAccessMode.DATABASE,
        start_date='2025-11-23',
        end_date='2025-11-24'
    )

    print(f'✓ Returned {len(df)} rows')
    print(f'✓ Columns: {df.columns.tolist()}')

    if not df.empty:
        print('\nSample data:')
        print(df.head(10).to_string())

        # Check format
        expected_cols = ['datetime', '공조기', '항목명', '값']
        if all(col in df.columns for col in expected_cols):
            print('\n✓ Correct column format')
        else:
            print(f'\n✗ Column format mismatch. Expected {expected_cols}')

except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('Test 2: load_oa_data (daily) with DATABASE mode')
print('=' * 60)

try:
    df = load_oa_data(
        mode=DataAccessMode.DATABASE,
        daily=True,
        start_date='2025-11-20',
        end_date='2025-11-25'
    )

    print(f'✓ Returned {len(df)} rows')
    print(f'✓ Columns: {df.columns.tolist()}')

    if not df.empty:
        print('\nSample data:')
        print(df.head().to_string())

        # Check format
        expected_cols = ['datetime', '외기온도', '외기습도']
        if all(col in df.columns for col in expected_cols):
            print('\n✓ Correct column format')
        else:
            print(f'\n✗ Column format mismatch. Expected {expected_cols}')

except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('Test 3: get_available_ahu_list with DATABASE mode')
print('=' * 60)

try:
    ahu_list = get_available_ahu_list(mode=DataAccessMode.DATABASE)
    print(f'✓ Returned {len(ahu_list)} AHUs')
    print(f'✓ First 5: {ahu_list[:5]}')

except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('✓ All database mode tests completed')
print('=' * 60)
