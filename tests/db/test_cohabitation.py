
import pytest
import sqlite3
import mlflow
import tempfile
import os
from pathlib import Path
from kladml.db import Project, init_db, session_scope
import kladml.db.session as db_session

# Use a real file for cohabitation test (mlflow needs a file uri or server)
@pytest.fixture
def shared_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    # Configure KladML to use this DB
    original_env = os.environ.get("KLADML_DB_PATH")
    os.environ["KLADML_DB_PATH"] = db_path
    
    # Reset internal engine to pick up new path
    db_session._engine = None
    db_session._session_factory = None
    
    init_db()
    
    yield db_path
    
    # Cleanup
    if original_env:
        os.environ["KLADML_DB_PATH"] = original_env
    else:
        del os.environ["KLADML_DB_PATH"]
        
    db_session._engine = None
    if os.path.exists(db_path):
        os.unlink(db_path)

def test_db_cohabitation(shared_db):
    """
    Test that KladML and MLflow can share the same SQLite file.
    """
    db_uri = f"sqlite:///{shared_db}"
    
    # 1. KladML writes tables
    with session_scope() as session:
        proj = Project(name="cohab-project", description="Living together")
        session.add(proj)
    
    # Verify KladML data physically exists
    conn = sqlite3.connect(shared_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM project WHERE name='cohab-project'")
    assert cursor.fetchone() is not None
    conn.close()
    
    # 2. MLflow writes tables to same DB
    mlflow.set_tracking_uri(db_uri)
    exp_id = mlflow.create_experiment("cohab-experiment")
    
    with mlflow.start_run(experiment_id=exp_id) as run:
        mlflow.log_param("test_param", "value")
    
    # 3. Verify COHABITATION
    conn = sqlite3.connect(shared_db)
    cursor = conn.cursor()
    
    # Check KladML table presence
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='project'")
    assert cursor.fetchone() is not None
    
    # Check MLflow table presence (MLflow creates 'experiments', 'runs', etc.)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'")
    assert cursor.fetchone() is not None
    
    # Check MLflow data
    cursor.execute("SELECT name FROM experiments WHERE name='cohab-experiment'")
    assert cursor.fetchone() is not None
    
    conn.close()
