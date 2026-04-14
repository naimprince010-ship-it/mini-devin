"""
Enterprise-oriented hooks (RBAC, future ticketing integrations).

Core Plodder runs without this package; import from ``mini_devin.enterprise``
when building multi-tenant or org-scoped deployments.
"""

from mini_devin.enterprise.rbac import Permission, Role, role_allows

__all__ = ["Permission", "Role", "role_allows"]
