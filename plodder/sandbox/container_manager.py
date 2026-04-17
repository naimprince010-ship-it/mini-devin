"""
Map RAG ``language_key`` slugs → Docker images; plan runs with pull hints + generic fallback.

Used by ``ExecutionSandbox.run_detected`` when ``language_key`` is supplied (e.g. from docs metadata).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

# Broad Debian image: git, build-essential, python3, ruby, many common build tools.
DEFAULT_GENERIC_IMAGE = "buildpack-deps:bookworm-scm"

# Doc slug / language_key → Docker image (override extension-based defaults)
LANGUAGE_KEY_IMAGES: dict[str, str] = {
    "python": "python:3.11-alpine",
    "py": "python:3.11-alpine",
    "javascript": "node:20-alpine",
    "js": "node:20-alpine",
    "node": "node:20-alpine",
    "typescript": "node:22-alpine",
    "ts": "node:22-alpine",
    "rust": "rust:alpine",
    "go": "golang:1.22-alpine",
    "cpp": "gcc:12-bookworm",
    "cplusplus": "gcc:12-bookworm",
    "c": "gcc:12-bookworm",
    "csharp": "mcr.microsoft.com/dotnet/sdk:8.0-alpine",
    "cs": "mcr.microsoft.com/dotnet/sdk:8.0-alpine",
    "fsharp": "mcr.microsoft.com/dotnet/sdk:8.0-alpine",
    "fs": "mcr.microsoft.com/dotnet/sdk:8.0-alpine",
    "ruby": "ruby:3.3-alpine",
    "php": "php:8.3-cli-alpine",
    "swift": "swift:5.10-jammy",
    "kotlin": "eclipse-temurin:21-jdk-alpine",
    "java": "eclipse-temurin:21-jdk-alpine",
    "scala": "sbtscala/scala-sbt:eclipse-temurin-jammy-21.0.2_13_1.9.9_1.0.0",
    "dart": "dart:stable",
    "lua": "nickblah/lua:5.4-alpine",
    "perl": "perl:5.38-slim",
    "haskell": "haskell:9.6-slim",
    "erlang": "erlang:26-alpine",
    "elixir": "elixir:1.16-alpine",
    "clojure": "clojure:temurin-21-tools-deps-alpine",
    "zig": "ziglang/zig:0.11.0-alpine",
    "nim": "nimlang/nim:2.0.2-alpine",
    "crystal": "crystallang/crystal:1.11-alpine",
    "julia": "julia:1.10-bookworm",
    "r": "rocker/r-ver:4.3.2",
    "bash": "alpine:3.19",
    "sh": "alpine:3.19",
    "shell": "alpine:3.19",
    "awk": "alpine:3.19",
    "docker": "docker:24-cli",
    "cmake": "kitware/cmake:3.28-debian",
    "ziglang": "ziglang/zig:0.11.0-alpine",
    "sql": "postgres:16-alpine",
}

# Slug → ``language_hint`` for ``build_toolchain_spec`` (supported runtimes only)
LANGUAGE_KEY_HINTS: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "node": "javascript",
    "cplusplus": "cpp",
    "sh": "shell",
    "bash": "shell",
    "cs": "csharp",
    "fs": "fsharp",
}

# Slugs that align with ``ToolchainSpec.language_id`` / ``infer`` hints (image override allowed)
_TOOLCHAIN_SLUGS: frozenset[str] = frozenset(
    {
        "python",
        "javascript",
        "typescript",
        "go",
        "rust",
        "shell",
        "c",
        "cpp",
        "java",
        "php",
        "csharp",
        "fsharp",
        "sql",
    }
)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower().strip()).strip("-") or "unknown"


def image_for_language_key(language_key: str | None) -> str | None:
    if not language_key or not str(language_key).strip():
        return None
    return LANGUAGE_KEY_IMAGES.get(_slug(str(language_key)))


def hint_for_language_key(language_key: str | None) -> str | None:
    """Runtime hint merged before extension inference (supported toolchains only)."""
    if not language_key:
        return None
    k = _slug(str(language_key))
    if k in LANGUAGE_KEY_HINTS:
        return LANGUAGE_KEY_HINTS[k]
    if k in _TOOLCHAIN_SLUGS:
        return k
    return None


def _slug_matches_toolchain(slug: str, language_id: str) -> bool:
    if slug == language_id:
        return True
    alias = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "node": "javascript",
        "bash": "shell",
        "sh": "shell",
        "cplusplus": "cpp",
        "cs": "csharp",
        "fs": "fsharp",
        "sql": "sql",
    }
    return alias.get(slug) == language_id


def generic_image() -> str:
    return (os.environ.get("PLODDER_GENERIC_IMAGE") or DEFAULT_GENERIC_IMAGE).strip()


def docker_image_exists(client: Any, image: str) -> bool:
    try:
        client.images.get(image)
        return True
    except Exception:
        return False


def pull_suggestion(image: str) -> str:
    return f"docker pull {image}"


@dataclass
class PlannedContainer:
    image: str
    argv: list[str]
    language_hint_used: str | None
    notes: list[str] = field(default_factory=list)
    used_generic_fallback: bool = False
    container_env: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def _generic_safe_runtime(language_id: str) -> bool:
    """Stacks likely usable on ``buildpack-deps`` without extra pulls."""
    return language_id in ("python", "shell", "javascript", "typescript", "cpp", "c")


def plan_container_run(
    *,
    entry: str,
    language_hint: str | None,
    language_key: str | None,
    python_image: str,
    node_image: str,
    go_image: str,
    rust_image: str,
    alpine_image: str,
    cpp_image: str,
    typescript_image: str = "node:22-alpine",
    java_image: str = "eclipse-temurin:21-jdk-alpine",
    php_image: str = "php:8.3-cli-alpine",
    dotnet_image: str = "mcr.microsoft.com/dotnet/sdk:8.0-alpine",
    maven_image: str = "maven:3.9.9-eclipse-temurin-21-alpine",
    gradle_image: str = "gradle:8.10.2-jdk21-alpine",
    composer_image: str = "composer:2",
    postgres_client_image: str = "postgres:16-alpine",
    sql_url: str | None = None,
    docker_client: Any | None = None,
    prefer_generic_if_image_missing: bool = True,
    auto_pull_missing: bool = False,
    pull_fn: Callable[[str], None] | None = None,
    workspace_files: dict[str, str] | None = None,
) -> PlannedContainer:
    from plodder.sandbox.toolchain_detect import build_toolchain_spec

    notes: list[str] = []
    lk = hint_for_language_key(language_key)
    merged = lk or (language_hint.strip() if language_hint and language_hint.strip() else None)
    if merged in ("", "auto"):
        merged = None

    spec = build_toolchain_spec(
        entry,
        language_hint=merged,
        python_image=python_image,
        node_image=node_image,
        go_image=go_image,
        rust_image=rust_image,
        alpine_image=alpine_image,
        cpp_image=cpp_image,
        typescript_image=typescript_image,
        java_image=java_image,
        php_image=php_image,
        dotnet_image=dotnet_image,
        maven_image=maven_image,
        gradle_image=gradle_image,
        composer_image=composer_image,
        postgres_client_image=postgres_client_image,
        workspace_files=workspace_files,
        sql_url=sql_url,
    )
    if spec.language_id == "rust" and workspace_files and any(
        k.replace("\\", "/").endswith("Cargo.toml") for k in workspace_files
    ):
        notes.append("Rust: using `cargo run` (set network:true if crates.io fetch fails).")
    if spec.language_id in ("csharp", "fsharp"):
        notes.append(
            "C#/F# uses `dotnet new` + `dotnet run` (NuGet). If restore fails, use sandbox_run with network:true."
        )
    if spec.language_id == "sql" and sql_url:
        notes.append("SQL: use network:true if psql cannot reach the database host from the sandbox.")
    if spec.language_id == "java" and workspace_files:
        keys = {k.replace("\\", "/").lstrip("/") for k in workspace_files}
        if any(k.endswith("/pom.xml") or k == "pom.xml" for k in keys):
            notes.append("Java/Maven: use network:true on first run if dependencies cannot be downloaded.")
        elif any(
            k.endswith("/build.gradle") or k.endswith("/build.gradle.kts") for k in keys
        ):
            notes.append("Java/Gradle: use network:true if wrapper or dependencies need to be fetched.")
    if spec.language_id == "php" and workspace_files:
        keys = {k.replace("\\", "/").lstrip("/") for k in workspace_files}
        if any(k.endswith("/composer.json") or k == "composer.json" for k in keys):
            notes.append("PHP/Composer: use network:true if packagist cannot be reached.")
    slug = _slug(str(language_key)) if language_key else ""
    mapped = image_for_language_key(language_key)
    image = spec.image
    if mapped and (
        lk is not None
        or (slug and _slug_matches_toolchain(slug, spec.language_id))
    ):
        image = mapped
    elif mapped and slug:
        notes.append(
            f"language_key {slug!r} does not match detected toolchain ({spec.language_id}); "
            f"keeping image {spec.image!r}. For stacks without auto-run, use `sandbox_shell` or add a matching entry file."
        )
    argv = list(spec.argv)
    used_generic = False

    if docker_client is not None and not docker_image_exists(docker_client, image):
        notes.append(f"Image not found locally: {image}. {pull_suggestion(image)}")
        if auto_pull_missing:
            try:
                if pull_fn:
                    pull_fn(image)
                else:
                    docker_client.images.pull(image)
                notes.append(f"Pulled: {image}")
            except Exception as e:  # noqa: BLE001
                notes.append(f"Auto-pull failed: {e}")

        still_missing = not docker_image_exists(docker_client, image)
        if still_missing and prefer_generic_if_image_missing and _generic_safe_runtime(spec.language_id):
            gen = generic_image()
            notes.append(
                f"Falling back to generic image {gen!r} (set PLODDER_GENERIC_IMAGE to override). "
                "Exotic compilers (e.g. rustc) are not assumed on this image."
            )
            image = gen
            used_generic = True

    return PlannedContainer(
        image=image,
        argv=argv,
        language_hint_used=merged,
        notes=notes,
        used_generic_fallback=used_generic,
        container_env=spec.container_env,
    )
