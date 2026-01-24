from datetime import datetime, timezone
from typing import Any
from enum import Enum
from sqlmodel import Field, SQLModel, Relationship
from sqlalchemy import Column, JSON

def utc_now():
    return datetime.now(timezone.utc)

# --- Enums ---

class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

class DataType(str, Enum):
    TABULAR = "tabular"
    TIMESERIES = "timeseries"
    IMAGE = "image"
    TEXT = "text"
    OTHER = "other"

# --- Models ---

class Project(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    
    # Relationships
    families: list["Family"] = Relationship(back_populates="project")

class Family(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: str | None = None
    project_id: int = Field(foreign_key="project.id")
    
    # Store experiment names as a JSON list of strings
    experiment_names: list[str] = Field(default=[], sa_column=Column(JSON))
    
    created_at: datetime = Field(default_factory=utc_now)
    
    # Relationships
    project: Project = Relationship(back_populates="families")
    
    # Methods for backward compatibility
    def add_experiment(self, experiment_name: str):
        if experiment_name not in self.experiment_names:
            # Create a new list to ensure SQLAlchemy detects the change
            self.experiment_names = self.experiment_names + [experiment_name]
            
    def remove_experiment(self, experiment_name: str):
        if experiment_name in self.experiment_names:
            self.experiment_names = [e for e in self.experiment_names if e != experiment_name]

class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    path: str
    description: str | None = None
    data_type: DataType = Field(default=DataType.OTHER)
    
    created_at: datetime = Field(default_factory=utc_now)

class RegistryArtifact(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    version: str = Field(index=True)
    artifact_type: str = Field(index=True) # model, preprocessor, dataset, etc.
    path: str # Path in registry
    run_id: str | None = Field(default=None, index=True)
    status: str = Field(default="production") # production, staging, archived
    
    # Metadata
    tags: list[str] = Field(default=[], sa_column=Column(JSON))
    metadata_json: dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
