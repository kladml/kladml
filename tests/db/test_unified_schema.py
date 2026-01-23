
import pytest
from sqlmodel import Session, SQLModel, create_engine, select
# Expecting these imports to fail initially
from kladml.db.models import Project, Run, RunStatus

@pytest.fixture(name="engine")
def fixture_engine():
    engine = create_engine("sqlite:///:memory:")
    # Only create tables if models exist
    if Project:
        SQLModel.metadata.create_all(engine)
    return engine

@pytest.fixture(name="session")
def fixture_session(engine):
    with Session(engine) as session:
        yield session

def test_models_exist():
    """Test that Project and Run models are defined."""
    assert Project is not None, "Project model not implemented"
    assert Run is not None, "Run model not implemented"

def test_create_project(session):
    """Test creating a project."""
    if not Project:
        pytest.fail("Project model not implemented")
        
    project = Project(name="test-project", description="A test project")
    session.add(project)
    session.commit()
    session.refresh(project)
    
    assert project.id is not None
    assert project.name == "test-project"

def test_create_run(session):
    """Test creating a run linked to a project."""
    if not Run or not Project:
        pytest.fail("Models not implemented")

    project = Project(name="test-project-2")
    session.add(project)
    session.commit()
    
    run = Run(
        project_id=project.id,
        experiment_name="exp-1",
        status=RunStatus.PENDING
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    
    assert run.id is not None
    assert run.project_id == project.id
    assert run.status == "pending"
    
    # Test relationship
    # This assumes we implement relationships, which we should
    # assert run.project.name == "test-project-2"
