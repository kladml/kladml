"""
Database module for KladML SDK.

Provides SQLite-based persistence for Projects only.
Experiments and runs are managed by MLflow.
"""

from kladml.db.models import Base, Project
from kladml.db.session import get_session, init_db, get_db_path, session_scope

__all__ = [
    "Base",
    "Project",
    "get_session",
    "init_db",
    "get_db_path",
    "session_scope",
]
