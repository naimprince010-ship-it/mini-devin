# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-04-14

### Added

- Event-driven backbone (`mini_devin/backbone/`): `EventStream`, `AgentStateStore`, `LocalRuntime`, `DockerRuntime`, `AgentController`, `AgentHostRuntime`.
- `run_simple` wires backbone setup/teardown; optional `PLODDER_BACKBONE_FOR_RUN` wraps full `Agent.run()` main phase in `try`/`finally`.
- Public SDK surface (`mini_devin.sdk`), `plodder serve`, starter enterprise RBAC (`mini_devin.enterprise`).
- `docs/PLATFORM.md`, backbone env vars in `.env.example`.
- CI job **Unit core** as a non-optional gate for backbone + enterprise unit tests.

### Changed

- CI **Build** no longer depends on **Eval** (eval remains on PRs but does not block packaging).

## [0.1.0]

Baseline before backbone release gate; see git history for prior features.
