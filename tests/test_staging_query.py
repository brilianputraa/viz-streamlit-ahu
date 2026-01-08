"""
Test querying staging table directly for sensor data.

This bypasses ahu_query_lib and queries ahu_readings_staging directly,
then transforms to the format expected by the visualization.
"""
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, '/Users/putra/viz-streamlit-ahu')

from db_config import get_database_connection_config
import psycopg2

# Column mapping: staging table → expected 항목명
COLUMN_MAPPING = {
    'cooling_coil_valve': 'CCV',
    'heating_coil_valve': 'HCV',
    'supply_fan_status': 'SFST',
    'supply_fan_status_1': 'SFST1',
    'supply_fan_status_2': 'SFST2',
    'exhaust_fan_status': 'EFST',
    'return_fan_status': 'RFST',
    'after_coil_supply_fan_status': 'AC_SFST',
    'pre_coil_supply_fan_status': 'PC_SFST',
    'oau_supply_fan_status': 'OAU_SFST',
    'compressor_status': 'COMP',
    'electric_heater_status': 'EH',
}

def fetch_sensor_data_from_staging(
    ahu_id: str,
    start_date: str = None,
    end_date: str = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch sensor data from staging table and transform to viz format.

    Returns DataFrame with columns: [datetime, 공조기, 항목명, 값]
    """
    config = get_database_connection_config()
    conn = psycopg2.connect(**config)

    # Select columns that exist in both mapping and table
    columns_to_fetch = [col for col in COLUMN_MAPPING.keys()]

    # Build dynamic query
    columns_sql = ', '.join(['timestamp', 'ahu_id'] + columns_to_fetch)
    where_clauses = ['ahu_id = %s']
    params = [ahu_id]

    if start_date:
        where_clauses.append('timestamp >= %s')
        params.append(start_date)

    if end_date:
        where_clauses.append('timestamp <= %s')
        params.append(end_date)

    query = f'''
        SELECT {columns_sql}
        FROM ahu_data.ahu_readings_staging
        WHERE {' AND '.join(where_clauses)}
        ORDER BY timestamp DESC
        LIMIT %s
    '''
    params.append(limit)

    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=['datetime', '공조기', '항목명', '값'])

    # Melt to long format
    id_vars = ['timestamp', 'ahu_id']
    value_vars = [col for col in columns_to_fetch if col in df.columns]

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='staging_column',
        value_name='값'
    )

    # Map staging column names to expected 항목명
    df_long['항목명'] = df_long['staging_column'].map(COLUMN_MAPPING)
    df_long = df_long.dropna(subset=['항목명', '값'])

    # Rename columns to match expected format
    df_long['공조기'] = df_long['ahu_id']
    df_long['datetime'] = pd.to_datetime(df_long['timestamp']).dt.tz_localize(None)

    return df_long[['datetime', '공조기', '항목명', '값']]

def fetch_outdoor_air_data(limit: int = 1000) -> pd.DataFrame:
    """
    Fetch outdoor air data from outdoor_weather table.

    Returns DataFrame with columns: [datetime, 외기온도, 외기습도]
    """
    config = get_database_connection_config()
    conn = psycopg2.connect(**config)

    query = '''
        SELECT timestamp, outdoor_temperature, outdoor_humidity
        FROM ahu_data.outdoor_weather
        ORDER BY timestamp DESC
        LIMIT %s
    '''

    df = pd.read_sql(query, conn, params=(limit,))
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=['datetime', '외기온도', '외기습도'])

    # Rename columns to Korean names expected by visualization
    df = df.rename(columns={
        'outdoor_temperature': '외기온도',
        'outdoor_humidity': '외기습도'
    })
    df['datetime'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    return df[['datetime', '외기온도', '외기습도']]


if __name__ == '__main__':
    print('=' * 60)
    print('Test 1: Fetch AHU01 Sensor Data')
    print('=' * 60)

    df_sensor = fetch_sensor_data_from_staging(
        ahu_id='AHU01',
        start_date='2025-11-23',
        limit=20
    )

    print(f'Fetched {len(df_sensor)} sensor readings')
    print('\nSample data:')
    print(df_sensor.head(10).to_string())

    print('\n' + '=' * 60)
    print('Test 2: Fetch Outdoor Air Data')
    print('=' * 60)

    df_oa = fetch_outdoor_air_data(limit=10)
    print(f'Fetched {len(df_oa)} outdoor readings')
    print('\nSample data:')
    print(df_oa.to_string())

    print('\n' + '=' * 60)
    print('Test 3: Check Data Completeness')
    print('=' * 60)

    config = get_database_connection_config()
    conn = psycopg2.connect(**config)
    cursor = conn.cursor()

    # Check all AHUs
    cursor.execute('''
        SELECT ahu_id, COUNT(*) as row_count
        FROM ahu_data.ahu_readings_staging
        GROUP BY ahu_id
        ORDER BY ahu_id
    ''')

    print('\nAHU Summary:')
    for row in cursor.fetchall():
        print(f"  {row[0]:10s}: {row[1]:>10,} rows")

    cursor.close()
    conn.close()

    print('\n✓ Direct staging query works!')
