"""
Map entry file / language hint → Docker image + argv for ``/workspace`` layout.

``files`` keys are relative POSIX paths (as uploaded to the sandbox tar).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ToolchainSpec:
    """Resolved execution plan inside the container."""

    language_id: str
    image: str
    argv: list[str]
    # Passed into ``docker create`` as ``environment`` (e.g. psql connection string).
    container_env: tuple[tuple[str, str], ...] = ()


def _posix(rel: str) -> str:
    return rel.replace("\\", "/")


def resolve_sql_url_from_env() -> str | None:
    """
    First non-empty ``SANDBOX_SQL_URL`` or ``DATABASE_URL`` suitable for ``psql`` (Postgres family).
    Skips obvious SQLite URLs so local ``DATABASE_URL=sqlite:///…`` does not trigger SQL mode.
    """
    for key in ("SANDBOX_SQL_URL", "DATABASE_URL"):
        raw = (os.environ.get(key) or "").strip()
        if not raw:
            continue
        scheme = raw.split("://", 1)[0].lower().split("+", 1)[0]
        if scheme == "sqlite":
            continue
        return raw
    return None


def _rust_cargo_argv(rel: str, workspace_files: dict[str, str] | None) -> list[str] | None:
    """If a Cargo workspace exists, run ``cargo run`` from that crate root; else None."""
    if not workspace_files:
        return None
    reln = _posix(rel).lstrip("/")
    manifests: list[str] = []
    for k in workspace_files:
        kn = _posix(k).lstrip("/")
        if kn == "Cargo.toml" or kn.endswith("/Cargo.toml"):
            manifests.append(kn)
    if not manifests:
        return None

    def _parent_len(m: str) -> int:
        p = str(PurePosixPath(m).parent)
        return 0 if p in (".", "") else len(p)

    matches: list[str] = []
    for m in manifests:
        parent = str(PurePosixPath(m).parent)
        prefix = "" if parent in (".", "") else parent.replace("\\", "/")
        if not prefix or reln == prefix or reln.startswith(prefix + "/"):
            matches.append(m)
    chosen = max(matches, key=_parent_len) if matches else manifests[0]

    crate_dir = str(PurePosixPath(chosen).parent)
    cwd = "/workspace" if crate_dir in (".", "") else f"/workspace/{crate_dir}"
    return ["sh", "-c", f"cd {cwd} && cargo run --quiet"]


def _workspace_key_set(workspace_files: dict[str, str] | None) -> set[str]:
    if not workspace_files:
        return set()
    return {_posix(k).lstrip("/") for k in workspace_files}


def _find_project_root(rel: str, keys: set[str], marker: str) -> str | None:
    """Directory prefix (``''`` for repo root) where ``marker`` exists as a snapshot key."""
    reln = _posix(rel).lstrip("/")
    p = PurePosixPath(reln)
    dirparts = p.parts[:-1] if p.parts else ()
    candidates: list[str] = [""]
    acc: list[str] = []
    for part in dirparts:
        acc.append(part)
        candidates.append("/".join(acc))
    for d in reversed(candidates):
        key = f"{d}/{marker}".strip("/") if d else marker
        if key in keys:
            return d
    return None


def infer_language_from_entry(entry: str, *, hint: str | None) -> str:
    if hint and hint.strip().lower() not in ("", "auto"):
        return hint.strip().lower()
    suf = PurePosixPath(entry).suffix.lower()
    if suf in (".py", ".pyw"):
        return "python"
    if suf in (".js", ".mjs", ".cjs"):
        return "javascript"
    if suf == ".ts":
        return "typescript"
    if suf == ".go":
        return "go"
    if suf == ".rs":
        return "rust"
    if suf in (".c",):
        return "c"
    if suf in (".cpp", ".cc", ".cxx", ".hpp"):
        return "cpp"
    if suf == ".sh":
        return "shell"
    if suf == ".java":
        return "java"
    if suf == ".php":
        return "php"
    if suf == ".cs":
        return "csharp"
    if suf == ".fs":
        return "fsharp"
    if suf == ".sql":
        return "sql"
    return "python"


def build_toolchain_spec(
    entry: str,
    *,
    language_hint: str | None,
    python_image: str,
    node_image: str,
    go_image: str = "golang:1.22-alpine",
    rust_image: str = "rust:alpine",
    alpine_image: str = "alpine:3.19",
    cpp_image: str = "gcc:12-bookworm",
    typescript_image: str = "node:22-alpine",
    java_image: str = "eclipse-temurin:21-jdk-alpine",
    php_image: str = "php:8.3-cli-alpine",
    dotnet_image: str = "mcr.microsoft.com/dotnet/sdk:8.0-alpine",
    maven_image: str = "maven:3.9.9-eclipse-temurin-21-alpine",
    gradle_image: str = "gradle:8.10.2-jdk21-alpine",
    composer_image: str = "composer:2",
    postgres_client_image: str = "postgres:16-alpine",
    workspace_files: dict[str, str] | None = None,
    sql_url: str | None = None,
) -> ToolchainSpec:
    rel = _posix(entry).lstrip("/")
    lang = infer_language_from_entry(rel, hint=language_hint)
    alias = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "node": "javascript",
        "cs": "csharp",
        "fs": "fsharp",
    }
    lang = alias.get(lang, lang)

    w = f"/workspace/{rel}"
    keys = _workspace_key_set(workspace_files)

    if lang == "sql":
        if not (sql_url or "").strip():
            return ToolchainSpec(
                "sql",
                alpine_image,
                [
                    "sh",
                    "-c",
                    "echo 'Set SANDBOX_SQL_URL or DATABASE_URL (Postgres) for .sql runs.' && exit 1",
                ],
            )
        sync = (sql_url or "").strip().replace("+asyncpg", "")
        if sync.startswith("postgres://"):
            sync = "postgresql://" + sync[len("postgres://") :]
        return ToolchainSpec(
            "sql",
            postgres_client_image,
            ["sh", "-c", f'psql "$PLODDER_SQL_URL" -v ON_ERROR_STOP=1 -f "{w}"'],
            container_env=(("PLODDER_SQL_URL", sync),),
        )

    if lang == "python":
        return ToolchainSpec("python", python_image, ["python", w])
    if lang in ("javascript", "js", "node"):
        return ToolchainSpec("javascript", node_image, ["node", w])
    if lang in ("typescript", "ts"):
        # Node 22+ can execute .ts with strip types (no npm needed for simple scripts).
        return ToolchainSpec("typescript", typescript_image, ["node", "--experimental-strip-types", w])
    if lang == "go":
        return ToolchainSpec("go", go_image, ["go", "run", w])
    if lang == "rust":
        cargo_argv = _rust_cargo_argv(rel, workspace_files)
        if cargo_argv is not None:
            return ToolchainSpec("rust", rust_image, cargo_argv)
        return ToolchainSpec(
            "rust",
            rust_image,
            ["sh", "-c", f"rustc {w} -o /tmp/a.out && /tmp/a.out"],
        )
    if lang == "shell":
        return ToolchainSpec("shell", alpine_image, ["sh", w])
    if lang == "c":
        return ToolchainSpec(
            "c",
            cpp_image,
            ["sh", "-c", f"gcc -O2 -Wall -o /tmp/a.out {w} && /tmp/a.out"],
        )
    if lang == "cpp":
        return ToolchainSpec(
            "cpp",
            cpp_image,
            ["sh", "-c", f"g++ -std=c++17 -O2 -Wall -o /tmp/a.out {w} && /tmp/a.out"],
        )
    if lang == "java":
        root = _find_project_root(rel, keys, "pom.xml")
        if root is not None:
            cwd = "/workspace" if not root else f"/workspace/{root}"
            return ToolchainSpec(
                "java",
                maven_image,
                ["sh", "-c", f"cd {cwd} && mvn -B -q -DskipTests package"],
            )
        root = _find_project_root(rel, keys, "build.gradle") or _find_project_root(
            rel, keys, "build.gradle.kts"
        )
        if root is not None:
            cwd = "/workspace" if not root else f"/workspace/{root}"
            gradlew_rel = f"{root}/gradlew".strip("/") if root else "gradlew"
            if gradlew_rel in keys:
                return ToolchainSpec(
                    "java",
                    gradle_image,
                    [
                        "sh",
                        "-c",
                        f"cd {cwd} && chmod +x gradlew 2>/dev/null; ./gradlew test --no-daemon -q",
                    ],
                )
            return ToolchainSpec(
                "java",
                gradle_image,
                ["sh", "-c", f"cd {cwd} && gradle test --no-daemon -q"],
            )
        p = PurePosixPath(rel)
        stem = p.stem
        parent = p.parent
        wdir = f"/workspace/{parent}".rstrip("/") if str(parent) not in (".", "") else "/workspace"
        return ToolchainSpec(
            "java",
            java_image,
            ["sh", "-c", f"javac {w} && java -cp {wdir} {stem}"],
        )
    if lang == "php":
        root = _find_project_root(rel, keys, "composer.json")
        if root is not None:
            cwd = "/workspace" if not root else f"/workspace/{root}"
            return ToolchainSpec(
                "php",
                composer_image,
                [
                    "sh",
                    "-c",
                    f"cd {cwd} && composer install --no-interaction --no-progress && php {w}",
                ],
            )
        return ToolchainSpec("php", php_image, ["php", w])
    if lang == "csharp":
        # First run hits NuGet — use sandbox_run with network:true when needed.
        return ToolchainSpec(
            "csharp",
            dotnet_image,
            [
                "sh",
                "-c",
                "rm -rf /tmp/plcsharp && mkdir -p /tmp/plcsharp && "
                "dotnet new console -o /tmp/plcsharp -n app --force && "
                f"cp {w} /tmp/plcsharp/Program.cs && dotnet run --project /tmp/plcsharp",
            ],
        )
    if lang == "fsharp":
        return ToolchainSpec(
            "fsharp",
            dotnet_image,
            [
                "sh",
                "-c",
                "rm -rf /tmp/plfsharp && mkdir -p /tmp/plfsharp && "
                "dotnet new console -lang F# -o /tmp/plfsharp -n app --force && "
                f"cp {w} /tmp/plfsharp/Program.fs && dotnet run --project /tmp/plfsharp",
            ],
        )
    # default
    return ToolchainSpec("python", python_image, ["python", w])


def pick_default_entry(files: dict[str, str]) -> str | None:
    """Choose a reasonable main file from the snapshot keys."""
    if not files:
        return None
    keys = sorted(files.keys(), key=lambda k: (k.count("/"), k))
    for prefer in (
        "main.py",
        "app.py",
        "index.js",
        "index.mjs",
        "main.ts",
        "src/main.py",
        "Program.cs",
        "Program.fs",
        "main.java",
        "index.php",
        "schema.sql",
        "migrate.sql",
    ):
        for k in keys:
            if k.replace("\\", "/").endswith(prefer):
                return k.replace("\\", "/")
    # first python or js
    for k in keys:
        low = k.lower()
        if low.endswith(
            (
                ".py",
                ".js",
                ".mjs",
                ".ts",
                ".go",
                ".rs",
                ".sh",
                ".c",
                ".cpp",
                ".cc",
                ".java",
                ".php",
                ".cs",
                ".fs",
                ".sql",
            )
        ):
            return k.replace("\\", "/")
    return keys[0].replace("\\", "/")


def image_for_shell_language(
    language: str | None,
    *,
    python_image: str,
    node_image: str,
    alpine_image: str,
    java_image: str = "eclipse-temurin:21-jdk-alpine",
    php_image: str = "php:8.3-cli-alpine",
    dotnet_image: str = "mcr.microsoft.com/dotnet/sdk:8.0-alpine",
    postgres_client_image: str = "postgres:16-alpine",
) -> str:
    """Pick a base image that likely contains toolchain binaries (npm, pip, sh)."""
    L = (language or "auto").lower()
    if L in ("sql", "postgres", "postgresql", "psql"):
        return postgres_client_image
    if L in ("javascript", "typescript", "node", "js", "ts"):
        return node_image
    if L in ("python", "py", "pip"):
        return python_image
    if L in ("java",):
        return java_image
    if L in ("php",):
        return php_image
    if L in ("csharp", "cs", "dotnet", "fsharp", "fs"):
        return dotnet_image
    return alpine_image
