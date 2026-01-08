# ahu_query_lib Integration Guide

This guide explains how to set up and integrate `ahu_query_lib` with viz-streamlit-ahu.

## Overview

`ahu_query_lib` is a Python library that provides structured access to AHU (Air Handling Unit) data from a PostgreSQL database. It sits between the database and the visualization application.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    viz-streamlit-ahu                        │
│                                                             │
│  ┌─────────────┐      ┌─────────────────────────────────┐ │
│  │  app2.py    │──────│      data_adapter.py            │ │
│  │  (UI Layer) │      │  (Unified Data Access Layer)    │ │
│  └─────────────┘      └─────────────────────────────────┘ │
│                                   │                        │
│                                   ▼                        │
│                        ┌──────────────────────┐           │
│                        │  ahu_query_lib       │           │
│                        │  (Query Interface)   │           │
│                        └──────────────────────┘           │
│                                   │                        │
└───────────────────────────────────┼────────────────────────┘
                                    ▼
                        ┌──────────────────────┐
                        │   PostgreSQL DB      │
                        │   - ahu_readings_    │
                        │     staging (4.7M)   │
                        │   - outdoor_weather  │
                        │   - energy_readings  │
                        └──────────────────────┘
```

## Prerequisites

### 1. Database Requirements

- PostgreSQL database running
- Database name: `ahu_monitoring`
- Tables:
  - `ahu_data.ahu_readings_staging` (4.7M+ rows of sensor data)
  - `ahu_data.outdoor_weather` (350K+ rows of weather data)
  - `ahu_data.energy_readings` (empty - requires ETL)

### 2. ahu-backend-server Repository

The `ahu_query_lib` is located in the `ahu-backend-server` repository:

```
ahu-backend-server/
└── ahu_query_lib/
    ├── __init__.py
    ├── core/
    │   ├── connection.py
    │   └── query_config.py
    ├── queries/
    │   ├── sensor.py
    │   └── energy.py
    └── builders/
        └── query_builder.py
```

## Installation Methods

### Method 1: PYTHONPATH (Recommended for Development)

This method uses the library directly from the source without installation.

```bash
# 1. Clone ahu-backend-server (if not already done)
cd /path/to/parent/directory
git clone <ahu-backend-server-repo-url>

# 2. Set PYTHONPATH
export PYTHONPATH=/path/to/ahu-backend-server:$PYTHONPATH

# 3. Verify installation
python3 -c "import ahu_query_lib as aql; print(aql.__version__)"
# Output should be: 0.1.0

# 4. Run the app
cd viz-streamlit-ahu
streamlit run app2.py
```

**Make PYTHONPATH permanent:**

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export PYTHONPATH=/Users/putra/ahu-backend-server:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### Method 2: Editable Install (Requires setup.py Fix)

This method installs the library in development mode.

```bash
# 1. Fix setup.py in ahu-backend-server (if needed)
cd /path/to/ahu-backend-server
# The setup.py needs to be fixed to install ahu_query_lib properly

# 2. Install in editable mode
pip install -e .

# 3. Verify installation
python3 -c "import ahu_query_lib as aql; print(aql.__version__)"
```

### Method 3: Conda Environment with PYTHONPATH

```bash
# 1. Create conda environment
conda env create -f environment.yml

# 2. Activate environment
conda activate viz-streamlit-ahu

# 3. The PYTHONPATH is already set in environment.yml
# Verify:
echo $PYTHONPATH

# 4. Run the app
streamlit run app2.py
```

## Configuration

### Database Connection

The database connection is configured via environment variables or `~/.pgpass` file:

**Option 1: Environment Variables (.env file)**
```bash
# .env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=ahu_monitoring
DB_USER=postgres
DB_PASSWORD=admin
```

**Option 2: ~/.pgpass file**
```bash
# ~/.pgpass format: hostname:port:database:username:password
localhost:5433:ahu_monitoring:postgres:admin
```

### Data Source Mode

Set the default data source in `.env`:
```bash
DATA_SOURCE_MODE=database  # or 'parquet'
```

## Usage Examples

### 1. Fetching Sensor Data

```python
import ahu_query_lib as aql

# Fetch raw sensor data for AHU01
df = aql.fetch_sensor_data(
    ahu_id='AHU01',
    start_date='2025-11-01',
    end_date='2025-11-30',
    aggregate='raw',  # or 'hourly', 'daily'
    limit=1000
)

print(df.columns)
# Output: ['timestamp', 'ahu_id', 'CCV', 'HCV', 'SFST', 'SAT', ...]
```

### 2. Fetching Energy Data

```python
# Fetch energy consumption and cost
df = aql.fetch_energy_consumption_cost(
    ahu_id=['AHU01', 'AHU02'],  # Multiple AHUs
    start_period='2025-11-01',
    end_period='2025-11-30',
    data_period='daily',
    detailed_cost=True,
    include_weather=True
)

print(df.columns)
# Output: ['ahu_id', 'date', 'cold_water_kwh', 'steam_kwh', ...]
```

### 3. Using data_adapter (Recommended)

```python
from data_adapter import DataAccessMode, load_ahu_detail

# Load sensor data
df = load_ahu_detail(
    ahu_name='AHU01',
    mode=DataAccessMode.DATABASE,
    start_date='2025-11-01',
    end_date='2025-11-30'
)

# Returns long format: [datetime, 공조기, 항목명, 값]
print(df.head())
```

## Troubleshooting

### Issue 1: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'ahu_query_lib'
```

**Solution:**
```bash
# Set PYTHONPATH
export PYTHONPATH=/path/to/ahu-backend-server:$PYTHONPATH

# Verify
python3 -c "import ahu_query_lib; print('OK')"
```

### Issue 2: SQL Syntax Error

**Error:**
```
syntax error at or near ")"
LINE 2:     SELECT timestamp, FIRST(10 *)
```

**Solution:**
This is a known bug in `ahu_query_lib/queries/sensor.py` line 78.

Apply the fix:
```python
# In ahu-backend-server/ahu_query_lib/queries/sensor.py
# Line 78 (before):
if limit:
    builder._select = builder._select[:1] + [f"FIRST({limit} *)"]

# Line 78 (after):
# Note: LIMIT is added to query string later, not in SELECT clause
```

### Issue 3: Database Connection Failed

**Error:**
```
psycopg2.OperationalError: connection refused
```

**Solution:**
```bash
# Check PostgreSQL is running
pgrep postgres

# Check database exists
psql -h localhost -p 5433 -U postgres -d ahu_monitoring -c "SELECT 1"

# Test connection
python3 -c "
from db_config import get_database_connection_config
import psycopg2
config = get_database_connection_config()
conn = psycopg2.connect(**config)
print('Connected!')
"
```

### Issue 4: Empty Energy Data

**Symptom:**
Energy queries return 0 rows.

**Explanation:**
The `energy_readings` table is empty. This is expected behavior.

**Workaround:**
Use parquet mode for energy data:
```python
# In app2.py sidebar, select "Parquet Files" mode
```

**Solution:**
Run ETL process to populate `energy_readings` from `ahu_readings_staging`.

## API Reference

### Main Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `fetch_sensor_data()` | Get raw sensor readings | DataFrame with parameter columns |
| `fetch_energy_consumption_cost()` | Get energy usage & costs | DataFrame with energy metrics |
| `fetch_daily_energy_summary()` | Get daily energy summary | DataFrame with daily aggregates |
| `fetch_all_ahu_metadata()` | Get all AHU information | Dict of AHU metadata |

### Parameter Mapping

`ahu_query_lib` uses parameter codes (not database column names):

| Parameter | Description | DB Column |
|-----------|-------------|-----------|
| `CCV` | Cooling Coil Valve | `cooling_coil_valve` |
| `HCV` | Heating Coil Valve | `heating_coil_valve` |
| `SFST` | Supply Fan Status | `supply_fan_status` |
| `SAT` | Supply Air Temperature | `supply_air_temperature` |
| `RAT` | Return Air Temperature | `return_air_temperature` |

## Development

### Running Tests

```bash
# Test ahu_query_lib directly
cd ahu-backend-server
python3 -m pytest tests/

# Test data_adapter integration
cd viz-streamlit-ahu
pytest tests/test_data_adapter.py -v
```

### Debugging

```python
import ahu_query_lib as aql

# Enable query logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run query and see SQL being executed
df = aql.fetch_sensor_data('AHU01', '2025-11-01', '2025-11-30', limit=10)
```

## Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| `ahu_query_lib` | 0.1.0 | Current version |
| PostgreSQL | 12+ | Tested on 14, 15, 16 |
| Python | 3.8+ | Tested on 3.12 |
| viz-streamlit-ahu | main | Latest commits |

## Related Files

- `data_adapter.py` - Unified data access layer
- `db_config.py` - Database configuration
- `environment.yml` - Conda environment setup
- `.env.example` - Environment variables template
