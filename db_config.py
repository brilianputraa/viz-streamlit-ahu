"""
Database configuration for viz-streamlit-ahu.

Allows users to configure database connection parameters.
"""
import os
from pathlib import Path
from typing import Dict, Any

# [추가됨] 로컬 Streamlit 기본 DB는 pgbouncer 사용 (connection storm 방지 + ahu_query_lib 기본값과 정합)
DEFAULT_CONFIG = {
    "host": "localhost",
    "port": 6432,
    "database": "ahu_read",
    "user": "postgres",
    "password": "admin",
}

# Environment variable mappings
DB_ENV_MAP = {
    "DB_HOST": "host",
    "DB_PORT": "port",
    "DB_NAME": "database",
    "DB_USER": "user",
    "DB_PASSWORD": "password"
}

# Pgbouncer environment variable mappings (takes precedence over DB_*)
PGB_ENV_MAP = {
    "PGB_HOST": "host",
    "PGB_PORT": "port",
    "PGB_NAME": "database",
    "PGB_USER": "user",
    "PGB_PASSWORD": "password"
}

# [추가됨] 호스트 실행 vs 도커 실행에 따라 pgbouncer host 자동 선택
def _default_pgbouncer_host() -> str:
    # Inside docker networks, "pgbouncer" is resolvable; on host, use localhost.
    return "pgbouncer" if os.path.exists("/.dockerenv") else "localhost"


# [추가됨] PGB_* 일부만 설정된 경우에도 동작하는 기본값
def _pgb_defaults() -> Dict[str, Any]:
    return {
        "host": _default_pgbouncer_host(),
        "port": 6432,
        "database": "ahu_read",
        "user": "postgres",
    }


def get_database_connection_config() -> Dict[str, Any]:
    """
    Get database connection configuration from environment or defaults.

    Precedence:
    1. If ANY PGB_* is set, use pgbouncer config (with sensible defaults)
    2. Otherwise, use DB_* values (existing behavior)
    3. If neither set, use DEFAULT_CONFIG for local development

    Returns:
        Dictionary with connection parameters (host, port, database, user, password)

    Example:
        >>> config = get_database_connection_config()
        >>> print(config)
        {'host': 'localhost', 'port': 5433, 'database': 'ahu_monitoring',
         'user': 'postgres', 'password': 'admin'}
    """
    # Check if any PGB_* variable is set
    pgb_vars_set = any(os.getenv(key) for key in PGB_ENV_MAP.keys())

    # [추가됨] Streamlit은 기본적으로 PgBouncer를 우선 사용 (DB_*가 localhost:5433 등으로 설정돼도 안전하게 라우팅)
    # - DB 직접 접속(5432/5433) 시 connection storm + 긴 쿼리로 Postgres가 쉽게 불안정해짐
    # - 필요 시 STREAMLIT_PREFER_PGBOUNCER=false 또는 STREAMLIT_FORCE_DIRECT_POSTGRES=true 로 비활성화 가능
    prefer_pgbouncer = os.getenv("STREAMLIT_PREFER_PGBOUNCER", "true").lower() == "true"
    force_direct = os.getenv("STREAMLIT_FORCE_DIRECT_POSTGRES", "false").lower() == "true"

    if pgb_vars_set or (prefer_pgbouncer and not force_direct):
        # Use pgbouncer configuration
        config = _pgb_defaults()

        # Override with PGB_* environment variables
        for env_key, config_key in PGB_ENV_MAP.items():
            value = os.getenv(env_key)
            if value:
                if config_key == "port":
                    config[config_key] = int(value)
                else:
                    config[config_key] = value

        # Special case: inherit password from DB_PASSWORD if PGB_PASSWORD not set
        if not os.getenv("PGB_PASSWORD"):
            config["password"] = os.getenv("DB_PASSWORD", DEFAULT_CONFIG["password"])

        return config
    else:
        # Use DB_* configuration (existing behavior)
        config = DEFAULT_CONFIG.copy()

        for env_key, config_key in DB_ENV_MAP.items():
            value = os.getenv(env_key)
            if value:
                if config_key == "port":
                    config[config_key] = int(value)
                else:
                    config[config_key] = value

        return config


def get_data_source_mode() -> str:
    """
    Get the current data source mode from environment.

    Returns:
        'parquet' or 'database'
    """
    return os.getenv("DATA_SOURCE_MODE", "parquet").lower()
