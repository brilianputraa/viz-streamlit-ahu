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
ENV_MAP = {
    "DB_HOST": "host",
    "DB_PORT": "port",
    "DB_NAME": "database",
    "DB_USER": "user",
    "DB_PASSWORD": "password"
}


def get_database_connection_config() -> Dict[str, Any]:
    """
    Get database connection configuration from environment or defaults.

    Returns:
        Dictionary with connection parameters (host, port, database, user, password)

    Example:
        >>> config = get_database_connection_config()
        >>> print(config)
        {'host': 'localhost', 'port': 5433, 'database': 'ahu_monitoring',
         'user': 'postgres', 'password': 'admin'}
    """
    config = DEFAULT_CONFIG.copy()

    # Override with environment variables if present
    for env_key, config_key in ENV_MAP.items():
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
