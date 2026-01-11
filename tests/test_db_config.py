import pytest
import os
from db_config import get_database_connection_config

def test_get_database_config():
    config = get_database_connection_config()
    assert "host" in config
    assert "port" in config
    assert "database" in config
    assert config["port"] == 6432

def test_pgb_vars_override_db_vars():
    """Test that PGB_* vars take precedence over DB_* vars."""
    # Save original values
    original_env = {}
    env_keys = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD',
                'PGB_HOST', 'PGB_PORT', 'PGB_NAME', 'PGB_USER', 'PGB_PASSWORD']
    for key in env_keys:
        if key in os.environ:
            original_env[key] = os.environ[key]

    try:
        # Set both PGB_* and DB_* vars
        os.environ["PGB_HOST"] = "pgbouncer"
        os.environ["PGB_PORT"] = "6432"
        os.environ["PGB_NAME"] = "ahu_read"
        os.environ["PGB_USER"] = "pgb_user"
        os.environ["PGB_PASSWORD"] = "pgb_pass"

        os.environ["DB_HOST"] = "direct_db"
        os.environ["DB_PORT"] = "5433"
        os.environ["DB_NAME"] = "ahu_monitoring"
        os.environ["DB_USER"] = "db_user"
        os.environ["DB_PASSWORD"] = "db_pass"

        config = get_database_connection_config()

        # Should use PGB_* values
        assert config["host"] == "pgbouncer"
        assert config["port"] == 6432
        assert config["database"] == "ahu_read"
        assert config["user"] == "pgb_user"
        assert config["password"] == "pgb_pass"
    finally:
        # Restore original environment
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]
        for key, value in original_env.items():
            os.environ[key] = value

def test_partial_pgb_vars_with_defaults():
    """Test that partial PGB_* configuration uses sensible defaults."""
    # Save original values
    original_env = {}
    env_keys = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD',
                'PGB_HOST', 'PGB_PORT', 'PGB_NAME', 'PGB_USER', 'PGB_PASSWORD']
    for key in env_keys:
        if key in os.environ:
            original_env[key] = os.environ[key]

    try:
        # Only set PGB_HOST (should trigger pgbouncer mode)
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]
        os.environ["PGB_HOST"] = "my-pgbouncer"

        config = get_database_connection_config()

        # Should use PGB_HOST + defaults for other PGB_* values
        assert config["host"] == "my-pgbouncer"
        assert config["port"] == 6432  # Default PGB_PORT
        assert config["database"] == "ahu_read"  # Default PGB_NAME
        assert config["user"] == "postgres"  # Default PGB_USER
    finally:
        # Restore original environment
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]
        for key, value in original_env.items():
            os.environ[key] = value

def test_single_pgb_var_triggers_pgb_mode():
    """Test that setting any single PGB_* var triggers pgbouncer mode."""
    # Save original values
    original_env = {}
    env_keys = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD',
                'PGB_HOST', 'PGB_PORT', 'PGB_NAME', 'PGB_USER', 'PGB_PASSWORD']
    for key in env_keys:
        if key in os.environ:
            original_env[key] = os.environ[key]

    try:
        # Clear all env vars
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]

        # Only set PGB_PORT
        os.environ["PGB_PORT"] = "7000"

        config = get_database_connection_config()

        # Should use PGB_PORT + other PGB_* defaults
        expected_default_host = "pgbouncer" if os.path.exists("/.dockerenv") else "localhost"
        assert config["host"] == expected_default_host  # Default PGB_HOST
        assert config["port"] == 7000  # From PGB_PORT
        assert config["database"] == "ahu_read"  # Default PGB_NAME
    finally:
        # Restore original environment
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]
        for key, value in original_env.items():
            os.environ[key] = value

def test_pgb_password_inherits_from_db_password():
    """Test that PGB_PASSWORD inherits from DB_PASSWORD if not set."""
    # Save original values
    original_env = {}
    env_keys = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD',
                'PGB_HOST', 'PGB_PORT', 'PGB_NAME', 'PGB_USER', 'PGB_PASSWORD']
    for key in env_keys:
        if key in os.environ:
            original_env[key] = os.environ[key]

    try:
        # Clear all env vars
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]

        os.environ["PGB_HOST"] = "pgbouncer"
        # Don't set PGB_PASSWORD
        os.environ["DB_PASSWORD"] = "inherited_password"

        config = get_database_connection_config()

        # Should inherit password from DB_PASSWORD
        assert config["password"] == "inherited_password"
    finally:
        # Restore original environment
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]
        for key, value in original_env.items():
            os.environ[key] = value

def test_db_vars_without_pgb_vars():
    """Test that DB_* vars work when no PGB_* vars are set."""
    # Save original values
    original_env = {}
    env_keys = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD',
                'PGB_HOST', 'PGB_PORT', 'PGB_NAME', 'PGB_USER', 'PGB_PASSWORD']
    for key in env_keys:
        if key in os.environ:
            original_env[key] = os.environ[key]

    try:
        # Clear all env vars
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]

        os.environ["DB_HOST"] = "db_host"
        os.environ["DB_PORT"] = "5432"
        os.environ["DB_NAME"] = "my_database"
        os.environ["STREAMLIT_PREFER_PGBOUNCER"] = "false"  # [추가됨] DB_* precedence test 용

        config = get_database_connection_config()

        # Should use DB_* values
        assert config["host"] == "db_host"
        assert config["port"] == 5432
        assert config["database"] == "my_database"
        assert config["user"] == "postgres"  # Default
        assert config["password"] == "admin"  # Default
    finally:
        # Restore original environment
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]
        for key, value in original_env.items():
            os.environ[key] = value
