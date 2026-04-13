"""
Temporary script: exercise SimpleDockerSandbox (mount, tools, file sync).
Run from mini-devin repo root: python test_sandbox_run.py
"""

from __future__ import annotations

import os
import sys

import docker.errors
from mini_devin.sandbox import SimpleDockerSandbox


def _posix_single_quote(s: str) -> str:
    """Shell single-quote for bash inside Linux container (host may be Windows)."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def main() -> int:
    root = os.path.abspath(os.getcwd())
    fname = "hello_plodder.txt"
    dummy = "hello from plodder sandbox test\n"

    print(f"project_root (host mount) -> /workspace: {root}")
    print(f"image: plodder-sandbox:latest\n")

    try:
        with SimpleDockerSandbox(root, image="plodder-sandbox:latest") as sb:
            print(f"container_id: {sb.container_id[:12]}…\n")

            for cmd in ("python3 --version", "git --version"):
                code, out = sb.exec_bash(cmd)
                text = out.decode(errors="replace") if isinstance(out, bytes) else str(out)
                print(f"$ {cmd}\nexit={code}\n{text}")
                if code != 0:
                    print(f"WARNING: non-zero exit for: {cmd}", file=sys.stderr)

            inner = (
                "open('/workspace/hello_plodder.txt','w',encoding='utf-8').write("
                + repr(dummy)
                + ")"
            )
            write_cmd = f"python3 -c {_posix_single_quote(inner)}"
            code, out = sb.exec_bash(write_cmd)
            wout = out.decode(errors="replace") if isinstance(out, bytes) else out
            print(f"$ (write {fname})\nexit={code}\n{wout!r}")
    except docker.errors.BuildError as exc:
        print(
            "BuildError while building plodder-sandbox (check Dockerfile.sandbox under "
            "repo root or mini_devin package tree):",
            exc,
            file=sys.stderr,
        )
        return 3
    except docker.errors.ImageNotFound as exc:
        print("ImageNotFound after pull/build:", exc, file=sys.stderr)
        return 3
    except docker.errors.DockerException as exc:
        print(
            "Docker daemon not reachable (start Docker Desktop / dockerd), or misconfigured:",
            exc,
            file=sys.stderr,
        )
        return 2

    host_path = os.path.join(root, fname)
    if os.path.isfile(host_path):
        print(f"\nOK: file visible on host: {host_path}")
        with open(host_path, encoding="utf-8") as f:
            got = f.read()
        print(f"host contents: {got!r}")
        if got != dummy:
            print("WARNING: host contents mismatch", file=sys.stderr)
        os.remove(host_path)
        print("(removed hello_plodder.txt from host)")
        return 0

    print(f"\nFAIL: expected file on host: {host_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
