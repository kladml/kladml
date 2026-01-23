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
TEST_DB_PATH = str(Path(TEST_DB_DIR) / "test_kladml.db")
os.environ["KLADML_DB_PATH"] = TEST_DB_PATH

from kladml.db import Project, Family, init_db
from kladml.db.session import session_scope, reset_db
import kladml.db.session as session_module

@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_dir():
    """Cleanup temporary directory after all tests."""
    yield
    shutil.rmtree(TEST_DB_DIR, ignore_errors=True)

@pytest.fixture(autouse=True)
def setup_test_db():
    """Reset database before each test."""
    # Force reset singleton engine to pick up test DB path
    session_module._engine = None
    session_module._session_factory = None
    
    # Now reset and init
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
            assert isinstance(project.id, int)
            # Relationship check
            assert len(project.families) == 0
    
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
            session.add(project)
            session.flush()
            
            session.flush()
            
            data = project.model_dump()
            assert data["name"] == "dict-test"
            assert data["description"] == "For dict test"
            # family_count is not in the model, handled by DTO layer
            assert "created_at" in data


class TestFamily:
    """Tests for Family model."""
    
    def test_create_family(self):
        """Test creating a family under a project."""
        with session_scope() as session:
            project = Project(name="family-project")
            session.add(project)
            session.flush()
            
            family = Family(name="test-family", project_id=project.id, description="Test family")
            session.add(family)
        
        with session_scope() as session:
            family = session.query(Family).filter_by(name="test-family").first()
            assert family is not None
            assert family.name == "test-family"
            assert family.experiment_names == []
    
    def test_add_experiments_to_family(self):
        """Test adding/removing experiments from a family."""
        with session_scope() as session:
            project = Project(name="exp-family-project")
            session.add(project)
            session.flush()
            
            family = Family(name="exp-family", project_id=project.id)
            session.add(family)
            session.flush()
            
            # Add experiments
            family.add_experiment("exp-1")
            family.add_experiment("exp-2")
            assert family.experiment_names == ["exp-1", "exp-2"]
            
            # Duplicate add should not add again
            family.add_experiment("exp-1")
            assert family.experiment_names == ["exp-1", "exp-2"]


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
