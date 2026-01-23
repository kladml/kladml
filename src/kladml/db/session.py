"""
Database session management for KladML SDK.

Provides SQLite connection management with:
- Automatic database initialization
- Session factory
- Path configuration
"""

import os
import logging
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from sqlmodel import SQLModel

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_DIR = Path.home() / ".kladml"
DEFAULT_DB_NAME = "kladml.db"


def get_db_path() -> Path:
    """
    Get the database file path.
    
    Uses KLADML_DB_PATH environment variable if set,
    otherwise defaults to ~/.kladml/kladml.db
    
    Returns:
        Path to the SQLite database file
    """
    env_path = os.environ.get("KLADML_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_DIR / DEFAULT_DB_NAME


def get_db_url() -> str:
    """
    Get the SQLAlchemy database URL.
    
    Returns:
        SQLite connection URL
    """
    db_path = get_db_path()
    return f"sqlite:///{db_path}"


# Lazy engine initialization
_engine = None
_session_factory = None


def _get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        db_path = get_db_path()
        
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        _engine = create_engine(
            get_db_url(),
            echo=False,  # Set to True for SQL debugging
            connect_args={"check_same_thread": False},  # SQLite specific
        )
        logger.debug(f"Created database engine: {db_path}")
    return _engine


def _get_session_factory():
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=_get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


def init_db() -> None:
    """
    Initialize the database.
    
    Creates all tables if they don't exist.
    Safe to call multiple times.
    """
    engine = _get_engine()
    SQLModel.metadata.create_all(bind=engine)
    logger.info(f"Database initialized: {get_db_path()}")


def get_session() -> Session:
    """
    Get a new database session.
    
    Returns:
        SQLAlchemy Session instance
        
    Note:
        Caller is responsible for closing the session.
        Prefer using session_scope() for automatic cleanup.
    """
    init_db()  # Ensure tables exist
    factory = _get_session_factory()
    return factory()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional scope around a series of operations.
    
    Automatically commits on success, rolls back on error,
    and closes the session.
    
    Yields:
        SQLAlchemy Session instance
        
    Example:
        with session_scope() as session:
            project = Project(name="my-project")
            session.add(project)
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_db() -> None:
    """
    Reset the database by dropping and recreating all tables.
    
    WARNING: This will delete all data!
    Use only for testing.
    """
    global _engine, _session_factory
    
    engine = _get_engine()
    SQLModel.metadata.drop_all(bind=engine)
    SQLModel.metadata.create_all(bind=engine)
    
    # Dispose engine to release file locks
    engine.dispose()
    
    # Reset cached instances
    _engine = None
    _session_factory = None
    
    logger.warning("Database reset complete - all data deleted")
