"""
Shared ProcessSandbox smoke test for Railway (no Docker).

Used by ``prod_test.py`` and the FastAPI ``/test-sandbox`` route. Logs to the
standard ``logging`` hierarchy so Railway log drains show each step.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mini_devin.sandbox.process_sandbox import ProcessSandbox

logger = logging.getLogger(__name__)

PROBE_FILENAME = "railway_process_sandbox_probe.txt"
PROBE_CONTENT = "process-sandbox-ok\n"


def _posix_single_quote(s: str) -> str:
    """Quote for ``bash -lc`` (host may be Windows; child is bash)."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def run_railway_process_sandbox_check(project_root: str | None = None) -> dict[str, Any]:
    """
    Run whoami / pwd / python3 --version, write a small file under the workspace, cat it.

    Returns a JSON-serializable dict with ``ok: bool`` and per-step details.
    """
    root = os.path.abspath(project_root or os.getcwd())
    host_path = os.path.join(root, PROBE_FILENAME)
    logger.info("[sandbox-test] starting ProcessSandbox check workspace=%s", root)
    print(f"[sandbox-test] workspace={root}", flush=True)

    result: dict[str, Any] = {
        "ok": False,
        "workspace": root,
        "session_api_key_present": False,
        "commands": [],
        "file_test": {},
        "errors": [],
    }

    try:
        sb = ProcessSandbox(root)
    except Exception as exc:
        msg = f"ProcessSandbox init failed: {exc}"
        logger.exception("[sandbox-test] %s", msg)
        print(f"[sandbox-test] ERROR {msg}", flush=True)
        result["errors"].append(msg)
        return result

    key = (sb.session_api_key or "").strip()
    result["session_api_key_present"] = bool(key)
    logger.info(
        "[sandbox-test] ProcessSandbox ready container_id=%s session_key_len=%s",
        sb.container_id,
        len(key),
    )
    print(
        f"[sandbox-test] ready container_id={sb.container_id} SESSION_API_KEY set={bool(key)}",
        flush=True,
    )

    for cmd in ("whoami", "pwd", "python3 --version"):
        code, raw = sb.exec_bash(cmd)
        text = raw.decode("utf-8", errors="replace").strip()
        entry = {"command": cmd, "exit_code": code, "output": text}
        result["commands"].append(entry)
        logger.info("[sandbox-test] cmd=%r exit=%s out=%r", cmd, code, text[:500])
        print(f"[sandbox-test] $ {cmd} -> exit={code} out={text!r}", flush=True)
        if code != 0:
            result["errors"].append(f"command failed: {cmd} (exit {code})")

    rel_path = PROBE_FILENAME
    inner = (
        "open("
        + repr(rel_path)
        + ", 'w', encoding='utf-8').write("
        + repr(PROBE_CONTENT)
        + ")"
    )
    write_cmd = f"python3 -c {_posix_single_quote(inner)} && cat {_posix_single_quote(rel_path)}"
    code, raw = sb.exec_bash(write_cmd)
    cat_text = raw.decode("utf-8", errors="replace")
    exists = os.path.isfile(host_path)
    result["file_test"] = {
        "relative_path": rel_path,
        "host_path": host_path,
        "host_file_exists": exists,
        "write_cat_exit_code": code,
        "cat_output": cat_text,
    }
    logger.info(
        "[sandbox-test] file probe exit=%s host_exists=%s cat_len=%s",
        code,
        exists,
        len(cat_text),
    )
    print(
        f"[sandbox-test] file probe exit={code} host_exists={exists} cat={cat_text!r}",
        flush=True,
    )
    if code != 0:
        result["errors"].append(f"write/cat failed (exit {code})")
    if not exists:
        result["errors"].append("probe file missing on host after write")
    if cat_text.strip() != PROBE_CONTENT.strip():
        result["errors"].append("cat output mismatch")

    result["ok"] = len(result["errors"]) == 0
    logger.info("[sandbox-test] finished ok=%s", result["ok"])
    print(f"[sandbox-test] finished ok={result['ok']}", flush=True)

    if os.path.isfile(host_path):
        try:
            os.remove(host_path)
            logger.info("[sandbox-test] removed probe file %s", host_path)
            print(f"[sandbox-test] removed {host_path}", flush=True)
        except OSError as exc:
            logger.warning("[sandbox-test] probe cleanup failed: %s", exc)

    return result


def allow_test_sandbox_http_route() -> bool:
    """Enable ``/test-sandbox`` on Railway by default; opt out with ``ALLOW_TEST_SANDBOX_ROUTE=0``."""
    flag = (os.getenv("ALLOW_TEST_SANDBOX_ROUTE") or "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    return bool((os.getenv("RAILWAY_ENVIRONMENT") or "").strip())
