import pytest
from db_config import get_database_connection_config

def test_get_database_config():
    config = get_database_connection_config()
    assert "host" in config
    assert "port" in config
    assert "database" in config
    assert config["port"] == 5433
