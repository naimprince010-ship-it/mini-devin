"""Tests for ``parse_git_remote_url``."""

import os

import pytest

from mini_devin.integrations.remote_parser import parse_git_remote_url


def test_github_https():
    p = parse_git_remote_url("https://github.com/acme/widget.git")
    assert p is not None
    assert p.platform == "github"
    assert p.path_with_namespace == "acme/widget"


def test_github_ssh():
    p = parse_git_remote_url("git@github.com:acme/widget.git")
    assert p is not None
    assert p.platform == "github"
    assert p.path_with_namespace == "acme/widget"


def test_gitlab_com_https():
    p = parse_git_remote_url("https://gitlab.com/group/sub/repo.git")
    assert p is not None
    assert p.platform == "gitlab"
    assert p.path_with_namespace == "group/sub/repo"
    assert p.api_v4_base == "https://gitlab.com/api/v4"


def test_gitlab_self_hosted_via_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GITLAB_API_URL", "https://git.example.org/api/v4")
    p = parse_git_remote_url("https://git.example.org/acme/app.git")
    assert p is not None
    assert p.platform == "gitlab"
    assert p.path_with_namespace == "acme/app"
    assert p.api_v4_base == "https://git.example.org/api/v4"
    monkeypatch.delenv("GITLAB_API_URL", raising=False)


def test_gitlab_host_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GITLAB_HOST", "code.internal")
    p = parse_git_remote_url("git@code.internal:grp/r.git")
    assert p is not None
    assert p.platform == "gitlab"
    assert p.api_v4_base == "https://code.internal/api/v4"
    monkeypatch.delenv("GITLAB_HOST", raising=False)


def test_bitbucket():
    p = parse_git_remote_url("https://bitbucket.org/acme/widget.git")
    assert p is not None
    assert p.platform == "bitbucket"


def test_unknown_host_without_env():
    p = parse_git_remote_url("https://sourcehut.org/~user/repo.git")
    assert p is not None
    assert p.platform == "unknown"
