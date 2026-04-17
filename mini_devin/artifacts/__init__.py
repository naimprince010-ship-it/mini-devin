"""
Artifacts module for Plodder.

Provides logging and persistence of task execution artifacts including
tool calls, file modifications, token usage, and final diffs.
"""

from .logger import ArtifactLogger, create_artifact_logger

__all__ = [
    "ArtifactLogger",
    "create_artifact_logger",
]
