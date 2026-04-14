#!/usr/bin/env sh
# Railway / Docker: run uvicorn as PID 1 so the HTTP listener is the main process.
# (Bootstrap watchdog uses a child process; some health probes and edges behave better with exec.)
set -eu
cd /app

export PYTHONPATH=".:${PYTHONPATH:-}"
PORT="${PORT:-8000}"

echo "[railway-entrypoint] cwd=$(pwd) PORT=${PORT}"

_run_alembic_bg() {
  if [ "${SKIP_ALEMBIC_UPGRADE:-}" = "1" ]; then
    echo "[railway-entrypoint] SKIP_ALEMBIC_UPGRADE=1 — skipping alembic"
    return
  fi
  case "${DATABASE_URL:-}" in
    *postgres*|*POSTGRES*) ;;
    *)
      echo "[railway-entrypoint] no Postgres DATABASE_URL — skipping alembic"
      return
      ;;
  esac
  echo "[railway-entrypoint] alembic upgrade head (background)"
  ( python -m alembic upgrade head && echo "[railway-entrypoint] alembic: ok" ) \
    || echo "[railway-entrypoint] alembic: failed (see logs); API is still up"
}

_run_alembic_bg &

echo "[railway-entrypoint] exec uvicorn on 0.0.0.0:${PORT}"
exec python -m uvicorn mini_devin.api:app --host 0.0.0.0 --port "${PORT}" --log-level info
