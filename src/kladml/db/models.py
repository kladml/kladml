from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from uuid import UUID, uuid4
from sqlmodel import Field, SQLModel, Relationship
from sqlalchemy import Column, JSON

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
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    families: List["Family"] = Relationship(back_populates="project")

class Family(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = None
    project_id: int = Field(foreign_key="project.id")
    
    # Store experiment names as a JSON list of strings
    experiment_names: List[str] = Field(default=[], sa_column=Column(JSON))
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
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
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    path: str
    description: Optional[str] = None
    data_type: DataType = Field(default=DataType.OTHER)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
