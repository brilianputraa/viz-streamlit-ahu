# viz-streamlit-ahu

공조기(AHU) 에너지 소모 및 비용 분석을 위한 Streamlit 대시보드

## Features

- 📊 **Parquet File Mode**: 로컬 parquet 파일을 통한 데이터 분석 (기본)
- 🗄️ **Database Mode**: 실시간 데이터베이스 연결을 통한 데이터 분석
- 📈 **에너지 소모 분석**: 냉수, 스팀, 전력 에너지 소모량 추적
- 💰 **비용 분석**: 공조기별/항목별 비용 분석
- 🌡️ **외기 데이터**: 외기 온도/습도 데이터와의 연관 분석
- 🤖 **GPT 인사이트**: OpenAI GPT를 활용한 데이터 분석 인사이트

## Prerequisites

- Python 3.8+
- PostgreSQL (for database mode)
- ahu-backend-server (for database mode)

## Installation

### 1. Clone the repository

```bash
cd /path/to/viz-streamlit-ahu
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables (optional)

```bash
cp .env.example .env
# Edit .env with your database credentials
```

## Running the Application

### Option 1: Parquet File Mode (Default)

```bash
streamlit run app2.py
```

### Option 2: Database Mode

First, set up the ahu_query_lib library:

```bash
# Add ahu-backend-server to PYTHONPATH
export PYTHONPATH=/path/to/ahu-backend-server:$PYTHONPATH

# Run the app
streamlit run app2.py
```

In the sidebar, select "Database" as the data source.

## Database Mode Setup

### Prerequisites

1. **ahu-backend-server** must be available
2. **PostgreSQL database** running with AHU data
3. **ahu_query_lib** installed and accessible

### Configuration

Create `.env` file:

```bash
DATA_SOURCE_MODE=database
DB_HOST=localhost
DB_PORT=5433
DB_NAME=ahu_monitoring
DB_USER=postgres
DB_PASSWORD=admin
```

### Troubleshooting

**Issue**: `ImportError: No module named 'ahu_query_lib'`

**Solution**:
```bash
export PYTHONPATH=/path/to/ahu-backend-server:$PYTHONPATH
```

**Issue**: Energy 데이터가 비어있습니다

**Solution**: This is expected. The `energy_readings` table is empty and requires ETL to populate from `ahu_readings_staging`. Sensor 데이터 (Detail view)는 정상 작동합니다.

## Data Source Comparison

| Feature | Parquet Mode | Database Mode |
|---------|--------------|---------------|
| **Sensor Data** | ✅ From parquet files | ✅ From `ahu_readings_staging` |
| **Outdoor Air Data** | ✅ From parquet files | ✅ From `outdoor_weather` |
| **Energy Data** | ✅ Pre-calculated | ⚠️ Requires ETL (`energy_readings` empty) |
| **Real-time Updates** | ❌ Requires manual reload | ✅ Direct DB queries |
| **Historical Data** | ✅ All available data | ✅ All available data |

## Project Structure

```
viz-streamlit-ahu/
├── app2.py                 # Main Streamlit application
├── data_adapter.py         # Unified data access layer
├── db_config.py           # Database configuration
├── loader.py              # Parquet file loader
├── common.py              # Common utilities
├── viz.py                 # Visualization functions
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── tests/                 # Test files
└── README.md              # This file
```

## API Reference

### data_adapter.py

#### `DataAccessMode`

Enum for data source selection:
- `DataAccessMode.PARQUET` - Use parquet files
- `DataAccessMode.DATABASE` - Use database

#### `load_ahu_detail(ahu_name, mode, start_date, end_date)`

Load detailed sensor data for a specific AHU.

**Returns**: DataFrame with columns `[datetime, 공조기, 항목명, 값]`

#### `load_oa_data(mode, daily, start_date, end_date)`

Load outdoor air data.

**Returns**: DataFrame with columns `[datetime, 외기온도, 외기습도]`

#### `get_available_ahu_list(mode)`

Get list of available AHU IDs.

**Returns**: List of AHU IDs (e.g., `['AHU01', 'AHU02', ...]`)

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_data_adapter.py -v

# Run with coverage
pytest --cov=. tests/
```

### Code Style

This project follows PEP 8 guidelines.

## Known Issues

1. **Energy Data Empty in Database Mode**: The `energy_readings` table is empty. Requires ETL process to populate from `ahu_readings_staging`.

2. **ahu_query_lib Bug**: There's a known SQL syntax bug in `sensor.py` line 78. If you encounter this error:
   ```
   syntax error at or near ")"
   ```
   Apply the fix in `/path/to/ahu-backend-server/ahu_query_lib/queries/sensor.py` line 78.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Submit a pull request

## License

[Specify your license here]

## Contact

[Add contact information]
