"""
LSP over stdio — JSON-RPC with Content-Length framing.

Works with ``typescript-language-server --stdio``, ``pylsp``, ``clangd``, etc.
Collects ``textDocument/publishDiagnostics`` after ``textDocument/didOpen``.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

Severity = Literal["error", "warning", "information", "hint", "unknown"]

_LSP_SEVERITY: dict[int, Severity] = {
    1: "error",
    2: "warning",
    3: "information",
    4: "hint",
}


def path_to_uri(path: Path) -> str:
    """Return a ``file://`` URI for LSP."""
    return path.resolve().as_uri()


def uri_to_path(uri: str) -> Path:
    """Best-effort map ``file://`` URI → local ``Path`` (Windows + POSIX)."""
    u = urlparse(uri)
    if u.scheme != "file":
        return Path(uri)
    path = unquote(u.path or "")
    # file:///C:/foo → path /C:/foo
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


@dataclass
class DiagnosticIssue:
    """Normalized diagnostic for ``self_heal`` (no sandbox run required)."""

    file_uri: str
    file_path: str
    severity: Severity
    code: str | None
    source: str | None
    message: str
    line: int
    character: int
    end_line: int
    end_character: int
    related_information: list[str] = field(default_factory=list)

    def to_compact_line(self) -> str:
        sev = self.severity if self.severity != "unknown" else "diag"
        loc = f"{self.line + 1}:{self.character + 1}"
        c = f" [{self.code}]" if self.code else ""
        return f"{sev.upper()} {loc}{c} {self.message}"


@dataclass
class DiagnosticsReport:
    issues: list[DiagnosticIssue]
    raw_lsp_messages: list[dict[str, Any]] = field(default_factory=list)

    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    def to_self_heal_prompt_block(self) -> str:
        """Inject into the repair LLM before sandbox execution."""
        if not self.issues:
            return ""
        lines = [
            "## Static analysis (LSP) — fix these before re-running tests/sandbox",
            "",
        ]
        by_file: dict[str, list[DiagnosticIssue]] = {}
        for issue in self.issues:
            by_file.setdefault(issue.file_path, []).append(issue)
        for fp, items in sorted(by_file.items()):
            lines.append(f"### `{fp}`")
            for it in sorted(items, key=lambda x: (x.line, x.character)):
                lines.append(f"- {it.to_compact_line()}")
            lines.append("")
        return "\n".join(lines).strip()


@dataclass
class LSPCommandConfig:
    """How to spawn the language server."""

    command: list[str]
    cwd: str | Path | None = None
    env: dict[str, str] | None = None


class _FramedJsonReader:
    """Accumulate bytes and peel off LSP Content-Length frames."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def append(self, chunk: bytes) -> list[dict[str, Any]]:
        self._buf.extend(chunk)
        out: list[dict[str, Any]] = []
        while True:
            sep = self._buf.find(b"\r\n\r\n")
            if sep < 0:
                break
            header_bytes = bytes(self._buf[:sep])
            rest = memoryview(self._buf)[sep + 4 :]
            length: int | None = None
            for line in header_bytes.decode("ascii", errors="replace").split("\r\n"):
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":", 1)[1].strip())
                    break
            if length is None:
                del self._buf[: sep + 4]
                continue
            if len(rest) < length:
                break
            body = bytes(rest[:length])
            del self._buf[: sep + 4 + length]
            try:
                out.append(json.loads(body.decode("utf-8")))
            except json.JSONDecodeError:
                continue
        return out


class StdioJsonRpcTransport:
    """JSON-RPC 2.0 over stdio (LSP framing)."""

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        self._proc = proc
        self._reader = _FramedJsonReader()
        self._incoming: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, name="lsp-stdio-read", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        while not self._stop.is_set():
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                break
            for msg in self._reader.append(chunk):
                self._dispatch(msg)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        # Response to our request
        if msg.get("id") is not None and ("result" in msg or "error" in msg):
            mid = msg.get("id")
            if isinstance(mid, int) and mid in self._pending:
                self._pending[mid].put(msg)
            return
        if "method" in msg:
            self._incoming.put(msg)

    def _write(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        with self._lock:
            assert self._proc.stdin is not None
            self._proc.stdin.write(header + body)
            self._proc.stdin.flush()

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def request(self, method: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            q: queue.Queue[dict[str, Any]] = queue.Queue()
            self._pending[rid] = q
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        try:
            resp = q.get(timeout=timeout)
        finally:
            self._pending.pop(rid, None)
        if "error" in resp:
            raise RuntimeError(f"LSP error: {resp['error']}")
        return resp.get("result")


def _parse_diagnostic(uri: str, d: dict[str, Any]) -> DiagnosticIssue:
    sev = _LSP_SEVERITY.get(int(d.get("severity", 0)), "unknown")
    rng = d.get("range") or {}
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    rel_info: list[str] = []
    for ri in d.get("relatedInformation") or []:
        loc = ri.get("location") or {}
        ru = loc.get("uri", "")
        rrng = loc.get("range") or {}
        rs = (rrng.get("start") or {})
        rel_info.append(f"{ru} L{int(rs.get('line', 0)) + 1}: {ri.get('message', '')}")
    code = d.get("code")
    if isinstance(code, dict):
        code = str(code.get("value", code))
    elif code is not None:
        code = str(code)
    return DiagnosticIssue(
        file_uri=uri,
        file_path=str(uri_to_path(uri)),
        severity=sev,
        code=code,
        source=d.get("source"),
        message=(d.get("message") or "").strip(),
        line=int(start.get("line", 0)),
        character=int(start.get("character", 0)),
        end_line=int(end.get("line", start.get("line", 0))),
        end_character=int(end.get("character", start.get("character", 0))),
        related_information=rel_info,
    )


def _drain_stderr(stream: Any) -> None:
    try:
        while True:
            line = stream.readline()
            if not line:
                break
    except Exception:
        pass


class LSPBridge:
    """
    High-level "eyes": spawn a language server, initialize, open a buffer, read diagnostics.
    """

    def __init__(self, config: LSPCommandConfig) -> None:
        self._config = config
        self._proc: subprocess.Popen[bytes] | None = None
        self._rpc: StdioJsonRpcTransport | None = None
        self._stderr_thread: threading.Thread | None = None
        self._initialized_root: str | None = None

    def start(self) -> None:
        if self._proc is not None:
            return
        cwd = str(self._config.cwd) if self._config.cwd else None
        env = {**os.environ, **(self._config.env or {})}
        self._proc = subprocess.Popen(
            self._config.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            bufsize=0,
        )
        if self._proc.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=_drain_stderr,
                args=(self._proc.stderr,),
                name="lsp-stderr-drain",
                daemon=True,
            )
            self._stderr_thread.start()
        self._rpc = StdioJsonRpcTransport(self._proc)
        self._rpc.start()

    def shutdown(self) -> None:
        if self._rpc is not None:
            try:
                self._rpc.request("shutdown", {}, timeout=5.0)
            except Exception:
                pass
            try:
                self._rpc.notify("exit", {})
            except Exception:
                pass
            self._rpc.stop()
            self._rpc = None
        self._proc = None
        self._initialized_root = None

    def __enter__(self) -> LSPBridge:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()

    def _initialize(self, root_uri: str, workspace_name: str = "workspace") -> None:
        assert self._rpc is not None
        if self._initialized_root is not None and self._initialized_root != root_uri:
            raise ValueError(
                "This LSPBridge was already initialized for a different workspace. "
                "Call shutdown() then start(), or use a separate LSPBridge per workspace root."
            )
        if self._initialized_root == root_uri:
            return
        caps: dict[str, Any] = {
            "workspace": {"workspaceFolders": True},
            "textDocument": {
                "synchronization": {
                    "dynamicRegistration": False,
                    "willSave": False,
                    "willSaveWaitUntil": False,
                    "didSave": False,
                },
                "publishDiagnostics": {"relatedInformation": True},
            },
        }
        params = {
            "processId": None,
            "rootUri": root_uri,
            "capabilities": caps,
            "workspaceFolders": [{"uri": root_uri, "name": workspace_name}],
            "clientInfo": {"name": "plodder-lsp-bridge", "version": "0.1"},
        }
        self._rpc.request("initialize", params, timeout=60.0)
        self._rpc.notify("initialized", {})
        self._initialized_root = root_uri

    def open_and_diagnose(
        self,
        *,
        workspace_root: Path,
        file_path: Path,
        source_text: str,
        language_id: str,
        wait_diagnostics_sec: float = 8.0,
    ) -> DiagnosticsReport:
        """
        ``language_id``: e.g. ``typescript``, ``python``, ``rust`` — must match what the server expects.
        """
        self.start()
        assert self._rpc is not None
        root = workspace_root.resolve()
        file_path = file_path.resolve()
        root_uri = path_to_uri(root)
        doc_uri = path_to_uri(file_path)

        self._initialize(root_uri)

        self._rpc.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": doc_uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": source_text,
                }
            },
        )

        issues: list[DiagnosticIssue] = []
        seen: set[tuple[int, int, str]] = set()
        raw: list[dict[str, Any]] = []
        deadline = time.monotonic() + wait_diagnostics_sec
        last_hit = time.monotonic()

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = self._rpc._incoming.get(timeout=min(0.35, max(0.05, remaining)))  # noqa: SLF001
            except queue.Empty:
                # no message for a while after we saw diagnostics → stop early
                if issues and (time.monotonic() - last_hit) > 0.5:
                    break
                continue
            raw.append(msg)
            if msg.get("method") != "textDocument/publishDiagnostics":
                continue
            params = msg.get("params") or {}
            if params.get("uri") != doc_uri:
                continue
            last_hit = time.monotonic()
            for d in params.get("diagnostics") or []:
                if not isinstance(d, dict):
                    continue
                issue = _parse_diagnostic(doc_uri, d)
                key = (issue.line, issue.character, issue.message)
                if key in seen:
                    continue
                seen.add(key)
                issues.append(issue)

        self._rpc.notify("textDocument/didClose", {"textDocument": {"uri": doc_uri}})
        return DiagnosticsReport(issues=issues, raw_lsp_messages=raw)
