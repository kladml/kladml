"""
Database session management for KladML SDK.

Provides SQLite connection management with:
- Automatic database initialization
- Session factory
- Path configuration
"""

from loguru import logger
from pathlib import Path
from collections.abc import Generator
from contextlib import contextmanager
from kladml.config.settings import settings

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, Session



# Default database location
DEFAULT_DB_DIR = Path.home() / ".kladml"
DEFAULT_DB_NAME = "kladml.db"


def get_db_path() -> Path:
    """
    Get the database file path from settings.
    Only valid for SQLite URLs.
    """
    url = settings.database_url
    if url.startswith("sqlite:///"):
        path_str = url.replace("sqlite:///", "")
        # If path is just a filename (no slash), put it in standardized db dir
        if "/" not in path_str and "\\" not in path_str:
            from kladml.utils.paths import get_root_data_path, DB_DIR
            return get_root_data_path() / DB_DIR / path_str
        return Path(path_str)
        
    # Fallback default
    from kladml.utils.paths import get_root_data_path, DB_DIR
    return get_root_data_path() / DB_DIR / "kladml.db"


def get_db_url() -> str:
    """
    Get the SQLAlchemy database URL.
    
    Returns:
        SQLite connection URL
    """
    return settings.database_url


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
            class_=Session,
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
