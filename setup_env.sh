#!/bin/bash
# Setup script for viz-streamlit-ahu environment.
# [추가됨] Streamlit 실행 전에 PYTHONPATH + pgbouncer(PGB_*)를 자동 세팅

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Export vars from repo root .env if present (Docker compose uses it; Streamlit doesn't by default).
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/.env"
  set +a
fi

# Ensure ahu_query_lib can be imported from the monorepo checkout.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Prefer pgbouncer for local Streamlit DB mode.
export PGB_HOST="${PGB_HOST:-localhost}"
export PGB_PORT="${PGB_PORT:-6432}"
export PGB_NAME="${PGB_NAME:-ahu_read}"
export PGB_USER="${PGB_USER:-postgres}"
# Inherit password if present in DB_PASSWORD, otherwise default "admin".
export PGB_PASSWORD="${PGB_PASSWORD:-${DB_PASSWORD:-admin}}"

echo "PYTHONPATH=${PYTHONPATH}"
echo "PGB_HOST=${PGB_HOST} PGB_PORT=${PGB_PORT} PGB_NAME=${PGB_NAME} PGB_USER=${PGB_USER}"
echo "You can now run: streamlit run app2.py"
