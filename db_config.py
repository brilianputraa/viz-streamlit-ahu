"""
Database configuration for viz-streamlit-ahu.

Allows users to configure database connection parameters.
"""
import os
from pathlib import Path
from typing import Dict, Any

# Default configuration for local development
DEFAULT_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "ahu_monitoring",
    "user": "postgres",
    "password": "admin"
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

# Sensible defaults for pgbouncer (when partially configured)
PGB_DEFAULTS = {
    "host": "pgbouncer",
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

    if pgb_vars_set:
        # Use pgbouncer configuration
        config = PGB_DEFAULTS.copy()

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
