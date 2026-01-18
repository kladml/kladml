"""
SQLAlchemy models for KladML SDK.

Contains Project and Family models - experiments and runs are managed by MLflow.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy import Column, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a short UUID for IDs."""
    return str(uuid.uuid4())[:8]


class Dataset(Base):
    """
    Dataset model.
    
    Represents a dataset stored in data/datasets/.
    
    Attributes:
        id: Unique identifier
        name: Dataset name (directory name)
        path: Relative path from workspace root
        description: Optional description
        created_at: Creation timestamp
    """
    __tablename__ = "local_datasets"
    
    id: str = Column(String(8), primary_key=True, default=generate_uuid)
    name: str = Column(String(255), unique=True, nullable=False, index=True)
    path: str = Column(String(512), nullable=False)
    description: Optional[str] = Column(Text, nullable=True)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Project(Base):
    """
    Project model.
    
    A project is a top-level container that groups related families.
    
    Hierarchy: Project > Family > Experiment > Run
    
    Attributes:
        id: Unique identifier (8-char UUID)
        name: Human-readable project name (unique)
        description: Optional project description
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = "projects"
    
    id: str = Column(String(8), primary_key=True, default=generate_uuid)
    name: str = Column(String(255), unique=True, nullable=False, index=True)
    description: Optional[str] = Column(Text, nullable=True)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    families = relationship("Family", back_populates="project", cascade="all, delete-orphan")
    
    @property
    def family_count(self) -> int:
        """Count of families in this project."""
        return len(self.families) if self.families else 0
    
    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name={self.name})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "family_count": len(self.families) if self.families else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Family(Base):
    """
    Family model.
    
    A family groups related experiments within a project by domain/purpose.
    Example: "glucose_forecasting", "driving_anomaly"
    
    Hierarchy: Project > Family > Experiment > Run
    
    Attributes:
        id: Unique identifier (8-char UUID)
        name: Family name (unique within project)
        project_id: FK to parent project
        description: Optional description
        experiment_names: List of MLflow experiment names
        created_at: Creation timestamp
    """
    __tablename__ = "families"
    
    id: str = Column(String(8), primary_key=True, default=generate_uuid)
    name: str = Column(String(255), nullable=False, index=True)
    project_id: str = Column(String(8), ForeignKey("projects.id"), nullable=False)
    description: Optional[str] = Column(Text, nullable=True)
    experiment_names: List[str] = Column(JSON, default=list)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    project = relationship("Project", back_populates="families")
    
    def __repr__(self) -> str:
        return f"<Family(id={self.id}, name={self.name}, project_id={self.project_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "description": self.description,
            "experiment_names": self.experiment_names or [],
            "experiment_count": len(self.experiment_names) if self.experiment_names else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    def add_experiment(self, experiment_name: str) -> None:
        """Add an MLflow experiment to this family."""
        if self.experiment_names is None:
            self.experiment_names = []
        if experiment_name not in self.experiment_names:
            self.experiment_names = self.experiment_names + [experiment_name]
    
    def remove_experiment(self, experiment_name: str) -> None:
        """Remove an MLflow experiment from this family."""
        if self.experiment_names and experiment_name in self.experiment_names:
            self.experiment_names = [e for e in self.experiment_names if e != experiment_name]

