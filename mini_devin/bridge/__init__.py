"""Local machine bridge: run terminal commands on the developer's PC while the agent runs in the cloud."""

from .manager import BridgeManager, get_bridge_manager

__all__ = ["BridgeManager", "get_bridge_manager"]
