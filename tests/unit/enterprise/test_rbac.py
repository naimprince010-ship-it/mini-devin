"""Tests for mini_devin.enterprise.rbac."""

from mini_devin.enterprise.rbac import Permission, Role, role_allows


def test_admin_has_all_permissions() -> None:
    for p in Permission:
        assert role_allows(Role.ADMIN, p)


def test_member_cannot_manage_users() -> None:
    assert role_allows(Role.MEMBER, Permission.RUN_AGENT)
    assert role_allows(Role.MEMBER, Permission.MANAGE_API_KEYS)
    assert not role_allows(Role.MEMBER, Permission.MANAGE_USERS)
    assert not role_allows(Role.MEMBER, Permission.MANAGE_INTEGRATIONS)


def test_viewer_read_only_sharing() -> None:
    assert role_allows(Role.VIEWER, Permission.SHARE_CONVERSATION)
    assert not role_allows(Role.VIEWER, Permission.RUN_AGENT)
