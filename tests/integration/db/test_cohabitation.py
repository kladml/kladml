
import pytest
import sqlite3
import mlflow
import tempfile
import os
import importlib

# Use a real file for cohabitation test (mlflow needs a file uri or server)
@pytest.fixture
def shared_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Set environment variable for this test
    os.environ["KLADML_DATABASE_URL"] = f"sqlite:///{db_path}"

    # Reload settings to pick up new env var
    import kladml.config.settings
    importlib.reload(kladml.config.settings)

    # Reload session module to pick up new settings
    import kladml.db.session
    kladml.db.session._engine = None
    kladml.db.session._session_factory = None
    importlib.reload(kladml.db.session)

    from kladml.db import init_db
    init_db()

    yield db_path

    # Cleanup
    if "KLADML_DATABASE_URL" in os.environ:
        del os.environ["KLADML_DATABASE_URL"]

    import kladml.db.session
    kladml.db.session._engine = None
    kladml.db.session._session_factory = None
    if os.path.exists(db_path):
        os.unlink(db_path)

def test_db_cohabitation(shared_db):
    """
    Test that KladML and MLflow can share the same SQLite file.
    """
    # Import after fixture has set up environment
    from kladml.db import Project, session_scope

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
