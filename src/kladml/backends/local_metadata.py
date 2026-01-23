"""
Local Metadata Backend (SQLite).

Implements MetadataInterface using SQLAlchemy and SQLite.
"""

from typing import List, Optional, Any
from kladml.interfaces.metadata import MetadataInterface, ProjectDTO, FamilyDTO, DatasetDTO
from kladml.db import Project, Family, init_db, session_scope

class LocalMetadata(MetadataInterface):
    """SQLite implementation of MetadataInterface."""
    
    def __init__(self):
        init_db()

    def _to_project_dto(self, p: Project) -> ProjectDTO:
        return ProjectDTO(
            id=str(p.id),
            name=p.name,
            description=p.description,
            created_at=p.created_at,
            updated_at=p.updated_at,
            family_count=len(p.families) if p.families else 0
        )

    def _to_family_dto(self, f: Family) -> FamilyDTO:
        return FamilyDTO(
            id=str(f.id),
            name=f.name,
            project_id=str(f.project_id),
            project_name=f.project.name if f.project else "?",
            description=f.description,
            experiment_names=f.experiment_names or [],
            created_at=f.created_at
        )

    def _to_dataset_dto(self, d: Any) -> DatasetDTO:
         # Using Any for d to avoid import circular issues if not clean, but normally it's available
         from kladml.db.models import Dataset
         return DatasetDTO(
             id=d.id,
             name=d.name,
             path=d.path,
             description=d.description,
             created_at=d.created_at
         )

    def create_dataset(self, name: str, path: str, description: Optional[str] = None) -> DatasetDTO:
        from kladml.db.models import Dataset
        with session_scope() as session:
            existing = session.query(Dataset).filter_by(name=name).first()
            if existing:
                return self._to_dataset_dto(existing)
            
            ds = Dataset(name=name, path=path, description=description)
            session.add(ds)
            session.flush()
            session.refresh(ds)
            return self._to_dataset_dto(ds)

    def list_datasets(self) -> List[DatasetDTO]:
        from kladml.db.models import Dataset
        with session_scope() as session:
            datasets = session.query(Dataset).order_by(Dataset.name).all()
            return [self._to_dataset_dto(d) for d in datasets]

    # Project Methods
    def create_project(self, name: str, description: Optional[str] = None) -> ProjectDTO:
        with session_scope() as session:
            existing = session.query(Project).filter_by(name=name).first()
            if existing:
                raise ValueError(f"Project '{name}' already exists")
            
            project = Project(name=name, description=description)
            session.add(project)
            session.flush()
            session.refresh(project)
            return self._to_project_dto(project)

    def get_project(self, name: str) -> Optional[ProjectDTO]:
        with session_scope() as session:
            project = session.query(Project).filter_by(name=name).first()
            if not project:
                return None
            return self._to_project_dto(project)

    def list_projects(self) -> List[ProjectDTO]:
        with session_scope() as session:
            projects = session.query(Project).order_by(Project.created_at.desc()).all()
            return [self._to_project_dto(p) for p in projects]

    def delete_project(self, name: str) -> None:
        with session_scope() as session:
            project = session.query(Project).filter_by(name=name).first()
            if not project:
                raise ValueError(f"Project '{name}' not found")
            session.delete(project)

    # Family Methods
    def create_family(self, name: str, project_name: str, description: Optional[str] = None) -> FamilyDTO:
        with session_scope() as session:
            project = session.query(Project).filter_by(name=project_name).first()
            if not project:
                raise ValueError(f"Project '{project_name}' not found")
            
            existing = session.query(Family).filter_by(project_id=project.id, name=name).first()
            if existing:
                raise ValueError(f"Family '{name}' already exists in project '{project_name}'")
            
            family = Family(name=name, project_id=project.id, description=description)
            session.add(family)
            session.flush()
            session.refresh(family)
            return self._to_family_dto(family)

    def get_family(self, name: str, project_name: str) -> Optional[FamilyDTO]:
        with session_scope() as session:
            project = session.query(Project).filter_by(name=project_name).first()
            if not project:
                return None
            
            family = session.query(Family).filter_by(project_id=project.id, name=name).first()
            if not family:
                return None
            return self._to_family_dto(family)

    def list_families(self, project_name: Optional[str] = None) -> List[FamilyDTO]:
        with session_scope() as session:
            query = session.query(Family)
            if project_name:
                project = session.query(Project).filter_by(name=project_name).first()
                if not project:
                    raise ValueError(f"Project '{project_name}' not found")
                query = query.filter_by(project_id=project.id)
            
            families = query.all()
            return [self._to_family_dto(f) for f in families]

    def delete_family(self, name: str, project_name: str) -> None:
        with session_scope() as session:
            project = session.query(Project).filter_by(name=project_name).first()
            if not project:
                raise ValueError(f"Project '{project_name}' not found")
                
            family = session.query(Family).filter_by(project_id=project.id, name=name).first()
            if not family:
                raise ValueError(f"Family '{name}' not found in project '{project_name}'")
                
            session.delete(family)

    def add_experiment_to_family(self, family_name: str, project_name: str, experiment_name: str) -> None:
        with session_scope() as session:
            project = session.query(Project).filter_by(name=project_name).first()
            if not project:
                raise ValueError(f"Project '{project_name}' not found")
            
            family = session.query(Family).filter_by(project_id=project.id, name=family_name).first()
            if not family:
                raise ValueError(f"Family '{family_name}' not found in project '{project_name}'")
            
            family.add_experiment(experiment_name)

    def remove_experiment_from_family(self, family_name: str, project_name: str, experiment_name: str) -> None:
        with session_scope() as session:
            project = session.query(Project).filter_by(name=project_name).first()
            if not project:
                raise ValueError(f"Project '{project_name}' not found")
            
            family = session.query(Family).filter_by(project_id=project.id, name=family_name).first()
            if not family:
                raise ValueError(f"Family '{family_name}' not found in project '{project_name}'")
            
            family.remove_experiment(experiment_name)
