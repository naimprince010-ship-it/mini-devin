"""Keyword → playbook tag inference (Agent._infer_auto_playbook_tags)."""

from mini_devin.orchestrator.agent import Agent


def test_review_triggers_code_review():
    assert Agent._infer_auto_playbook_tags("Please Review the PR") == ["code_review"]


def test_check_not_checkout():
    assert Agent._infer_auto_playbook_tags("Please check the build") == ["code_review"]
    assert Agent._infer_auto_playbook_tags("git checkout main") == []


def test_refactor_tag():
    assert "refactor" in Agent._infer_auto_playbook_tags("We need to refactor the module")


def test_order_and_dedupe():
    t = Agent._infer_auto_playbook_tags("Review and refactor the service")
    assert t == ["code_review", "refactor"]
