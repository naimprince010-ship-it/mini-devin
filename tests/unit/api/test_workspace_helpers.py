from mini_devin.api.app import _workspace_slug_from_git_url


def test_workspace_slug_from_https_git_url():
    assert _workspace_slug_from_git_url("https://github.com/octocat/Hello-World.git") == "hello-world"


def test_workspace_slug_from_ssh_git_url():
    assert _workspace_slug_from_git_url("git@github.com:naimprince010/ship-it.git") == "ship-it"


def test_workspace_slug_falls_back_for_empty_url():
    assert _workspace_slug_from_git_url("") == "repo"
