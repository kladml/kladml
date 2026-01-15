"""
Unit tests for KladML database layer.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path

# Set test database path before importing
TEST_DB_DIR = tempfile.mkdtemp()
os.environ["KLADML_DB_PATH"] = str(Path(TEST_DB_DIR) / "test_kladml.db")

from kladml.db import Project, init_db
from kladml.db.session import session_scope, reset_db

@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_dir():
    """Cleanup temporary directory after all tests."""
    yield
    shutil.rmtree(TEST_DB_DIR)

@pytest.fixture(autouse=True)
def setup_test_db():
    """Reset database before each test."""
    reset_db()
    init_db()
    yield
    # Cleanup after tests
    reset_db()


class TestProject:
    """Tests for Project model."""
    
    def test_create_project(self):
        """Test creating a project."""
        with session_scope() as session:
            project = Project(name="test-project", description="Test description")
            session.add(project)
        
        with session_scope() as session:
            project = session.query(Project).filter_by(name="test-project").first()
            assert project is not None
            assert project.name == "test-project"
            assert project.description == "Test description"
            assert project.id is not None
            assert len(project.id) == 8
            assert project.experiment_names == []  # Default empty list
    
    def test_project_unique_name(self):
        """Test that project names must be unique."""
        with session_scope() as session:
            project1 = Project(name="unique-project")
            session.add(project1)
        
        with pytest.raises(Exception):
            with session_scope() as session:
                project2 = Project(name="unique-project")
                session.add(project2)
    
    def test_project_to_dict(self):
        """Test project serialization."""
        with session_scope() as session:
            project = Project(name="dict-test", description="For dict test")
            project.experiment_names = ["exp1", "exp2"]
            session.add(project)
            session.flush()
            
            data = project.to_dict()
            assert data["name"] == "dict-test"
            assert data["description"] == "For dict test"
            assert data["experiment_names"] == ["exp1", "exp2"]
            assert data["experiment_count"] == 2
            assert "created_at" in data

    def test_add_remove_experiments(self):
        """Test adding/removing experiments from project."""
        with session_scope() as session:
            project = Project(name="manage-exp-project")
            session.add(project)
            session.flush()
            
            # Add
            project.add_experiment("exp-1")
            project.add_experiment("exp-2")
            assert project.experiment_names == ["exp-1", "exp-2"]
            
            # Remove
            project.remove_experiment("exp-1")
            assert project.experiment_names == ["exp-2"]
            
            # Remove non-existent (should not fail)
            project.remove_experiment("does-not-exist")
            assert project.experiment_names == ["exp-2"]


class TestSessionManagement:
    """Tests for session management."""
    
    def test_session_scope_commit(self):
        """Test that session_scope commits on success."""
        with session_scope() as session:
            project = Project(name="commit-test")
            session.add(project)
        
        # Should be persisted
        with session_scope() as session:
            project = session.query(Project).filter_by(name="commit-test").first()
            assert project is not None
    
    def test_session_scope_rollback(self):
        """Test that session_scope rolls back on error."""
        try:
            with session_scope() as session:
                project = Project(name="rollback-test")
                session.add(project)
                raise ValueError("Intentional error")
        except ValueError:
            pass
        
        # Should not be persisted
        with session_scope() as session:
            project = session.query(Project).filter_by(name="rollback-test").first()
            assert project is None
