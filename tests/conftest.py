import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# [추가됨] monorepo(ahu-backend-server) 체크아웃에서 ahu_query_lib를 로컬로 임포트 가능하게 설정
MONOREPO_ROOT = PROJECT_ROOT.parent
if (MONOREPO_ROOT / "ahu_query_lib").is_dir() and str(MONOREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MONOREPO_ROOT))

# [추가됨] 테스트 환경에서 DB 연결을 PgBouncer로 고정 (환경변수 누수/로컬 설정 차이로 인한 불안정 방지)
DEFAULT_PGB_HOST = "pgbouncer" if os.path.exists("/.dockerenv") else "localhost"
os.environ.setdefault("PGB_HOST", DEFAULT_PGB_HOST)
os.environ.setdefault("PGB_PORT", "6432")
os.environ.setdefault("PGB_NAME", "ahu_read")
os.environ.setdefault("PGB_USER", "postgres")
os.environ.setdefault("PGB_PASSWORD", os.getenv("DB_PASSWORD", "admin"))
