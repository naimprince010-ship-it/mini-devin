from __future__ import annotations

from mini_devin.reliability.deploy_ops import (
    evaluate_rollback_guard,
    run_deploy_preflight,
    validate_dependency_graph,
    validate_startup_sequence,
)


def test_dependency_graph_failure_missing_node() -> None:
    issues = validate_dependency_graph({"api": ["missing"], "database": []})
    assert any(item.code == "dependency_graph.missing_node" for item in issues)


def test_dependency_graph_failure_cycle_detected() -> None:
    issues = validate_dependency_graph({"api": ["database"], "database": ["api"]})
    assert any(item.code == "dependency_graph.cycle_detected" for item in issues)


def test_startup_ordering_violation_detected() -> None:
    issues = validate_startup_sequence(["boot.begin", "db.init.complete", "db.init.start"])
    assert any(item.code == "startup.sequence.order_violation" for item in issues)


def test_rollback_guard_behavior_blocked_when_strict(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PLODDER_ROLLBACK_GUARDS", "1")
    monkeypatch.setattr("mini_devin.reliability.deploy_ops.shutil.which", lambda _: None)

    report = evaluate_rollback_guard(tmp_path)

    assert report.safe_to_deploy is False
    assert any(check["name"] == "git_checkpoint_capable" and check["ok"] is False for check in report.checks)


def test_deploy_validation_failures_structured(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PLODDER_QUEUE_FAILOVER_POLICY", "not-valid")

    report = run_deploy_preflight(repo_root=tmp_path, startup_stage_history=["boot.begin"])
    payload = report.to_dict()

    assert payload["status"] == "failed"
    assert any(item["code"] == "config.invalid.queue_failover_policy" for item in payload["issues"])
