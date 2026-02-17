"""Database models for KladML experiment tracking.

This module defines SQLModel-based database models for persisting
ML experiment metadata, including projects, families, datasets, and
registered artifacts.

Example:
    >>> from kladml.db.models import Project, Family
    >>> project = Project(name="my_project", description="ML experiments")
    >>> family = Family(name="v1_experiments", project_id=project.id)
"""

from datetime import datetime, timezone
from typing import Any
from enum import Enum
from sqlmodel import Field, SQLModel, Relationship
from sqlalchemy import Column, JSON


def utc_now():
    """Return current UTC datetime with timezone info.

    Used as default factory for timestamp fields in database models.

    Returns:
        datetime: Current UTC time with timezone awareness.
    """
    return datetime.now(timezone.utc)

# --- Enums ---


class RunStatus(str, Enum):
    """Enumeration of possible run states for ML experiments.

    Attributes:
        PENDING: Run is queued but not started.
        RUNNING: Run is currently executing.
        COMPLETED: Run finished successfully.
        FAILED: Run encountered an error.
        STOPPED: Run was manually stopped.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class DataType(str, Enum):
    """Enumeration of supported data types for datasets.

    Attributes:
        TABULAR: Structured tabular data (CSV, Parquet, etc.).
        TIMESERIES: Time-series data with temporal ordering.
        IMAGE: Image data for computer vision tasks.
        TEXT: Text data for NLP tasks.
        OTHER: Other or unspecified data types.
    """

    TABULAR = "tabular"
    TIMESERIES = "timeseries"
    IMAGE = "image"
    TEXT = "text"
    OTHER = "other"

# --- Models ---


class Project(SQLModel, table=True):
    """Top-level container for organizing ML experiments.

    Projects group related experiment families together and provide
    a namespace for organizing ML work.

    Attributes:
        id: Primary key (auto-generated).
        name: Unique project name (indexed).
        description: Optional project description.
        created_at: UTC timestamp of project creation.
        updated_at: UTC timestamp of last update.
        families: List of experiment families in this project.
    """

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    families: list["Family"] = Relationship(back_populates="project")


class Family(SQLModel, table=True):
    """Group of related experiments within a project.

    Families organize experiments that share a common goal or
    configuration variant (e.g., hyperparameter tuning runs).

    Attributes:
        id: Primary key (auto-generated).
        name: Family name (indexed).
        description: Optional family description.
        project_id: Foreign key to parent project.
        experiment_names: List of experiment names in this family.
        created_at: UTC timestamp of family creation.
        project: Parent project relationship.
    """

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: str | None = None
    project_id: int = Field(foreign_key="project.id")

    # Store experiment names as a JSON list of strings
    experiment_names: list[str] = Field(default=[], sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=utc_now)

    # Relationships
    project: Project = Relationship(back_populates="families")

    def add_experiment(self, experiment_name: str):
        """Add an experiment to this family.

        Args:
            experiment_name: Name of the experiment to add.
        """
        if experiment_name not in self.experiment_names:
            # Create a new list to ensure SQLAlchemy detects the change
            self.experiment_names = self.experiment_names + [experiment_name]

    def remove_experiment(self, experiment_name: str):
        """Remove an experiment from this family.

        Args:
            experiment_name: Name of the experiment to remove.
        """
        if experiment_name in self.experiment_names:
            self.experiment_names = [e for e in self.experiment_names if e != experiment_name]


class Dataset(SQLModel, table=True):
    """Registered dataset for ML experiments.

    Tracks datasets used in experiments with their storage location
    and type information.

    Attributes:
        id: Primary key (auto-generated).
        name: Unique dataset name (indexed).
        path: File system path to the dataset.
        description: Optional dataset description.
        data_type: Type of data (tabular, timeseries, etc.).
        created_at: UTC timestamp of dataset registration.
    """

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    path: str
    description: str | None = None
    data_type: DataType = Field(default=DataType.OTHER)

    created_at: datetime = Field(default_factory=utc_now)


class RegistryArtifact(SQLModel, table=True):
    """Registered ML artifact (model, preprocessor, etc.).

    Tracks versioned artifacts in the model registry with metadata
    and status information for deployment management.

    Attributes:
        id: Primary key (auto-generated).
        name: Artifact name (indexed).
        version: Semantic version string (indexed).
        artifact_type: Type of artifact (model, preprocessor, dataset).
        path: File system path in the registry.
        run_id: Optional ID of the run that produced this artifact.
        status: Deployment status (production, staging, archived).
        tags: List of tags for categorization.
        metadata_json: Additional metadata as JSON.
        created_at: UTC timestamp of artifact registration.
        updated_at: UTC timestamp of last update.
    """

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    version: str = Field(index=True)
    artifact_type: str = Field(index=True)  # model, preprocessor, dataset, etc.
    path: str  # Path in registry
    run_id: str | None = Field(default=None, index=True)
    status: str = Field(default="production")  # production, staging, archived

    # Metadata
    tags: list[str] = Field(default=[], sa_column=Column(JSON))
    metadata_json: dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
