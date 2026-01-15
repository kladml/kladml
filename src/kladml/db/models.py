"""
SQLAlchemy models for KladML SDK.

Only contains Project model - experiments and runs are managed by MLflow.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy import Column, String, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a short UUID for IDs."""
    return str(uuid.uuid4())[:8]


class Project(Base):
    """
    Project model.
    
    A project is a container that groups related MLflow experiments.
    This is the only entity we store locally - experiments and runs
    are managed entirely by MLflow.
    
    Attributes:
        id: Unique identifier (8-char UUID)
        name: Human-readable project name (unique)
        description: Optional project description
        experiment_names: List of MLflow experiment names in this project
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = "projects"
    
    id: str = Column(String(8), primary_key=True, default=generate_uuid)
    name: str = Column(String(255), unique=True, nullable=False, index=True)
    description: Optional[str] = Column(Text, nullable=True)
    experiment_names: List[str] = Column(JSON, default=list)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name={self.name})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "experiment_names": self.experiment_names or [],
            "experiment_count": len(self.experiment_names) if self.experiment_names else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def add_experiment(self, experiment_name: str) -> None:
        """Add an MLflow experiment to this project."""
        if self.experiment_names is None:
            self.experiment_names = []
        if experiment_name not in self.experiment_names:
            self.experiment_names = self.experiment_names + [experiment_name]
    
    def remove_experiment(self, experiment_name: str) -> None:
        """Remove an MLflow experiment from this project."""
        if self.experiment_names and experiment_name in self.experiment_names:
            self.experiment_names = [e for e in self.experiment_names if e != experiment_name]
