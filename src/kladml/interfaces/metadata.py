"""
Metadata Interface.

Defines the contract for managing metadata entities (Projects, Families).
This allows swapping the backend (e.g., SQLite vs Postgres) without changing CLI logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ProjectDTO:
    id: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    family_count: int

@dataclass
class FamilyDTO:
    id: str
    name: str
    project_id: str
    project_name: str
    description: str | None
    experiment_names: list[str]
    created_at: datetime

@dataclass
class DatasetDTO:
    id: str
    name: str
    path: str
    description: str | None
    created_at: datetime

class MetadataInterface(ABC):
    """Abstract interface for metadata management."""
    
    @abstractmethod
    def create_dataset(self, name: str, path: str, description: str | None = None) -> DatasetDTO:
        """Create a new dataset."""
        pass

    @abstractmethod
    def list_datasets(self) -> list[DatasetDTO]:
        """List all datasets."""
        pass
    
    # Project Methods
    @abstractmethod
    def create_project(self, name: str, description: str | None = None) -> ProjectDTO:
        """Create a new project."""
        pass
        
    @abstractmethod
    def get_project(self, name: str) -> ProjectDTO | None:
        """Get a project by name."""
        pass
        
    @abstractmethod
    def list_projects(self) -> list[ProjectDTO]:
        """List all projects."""
        pass
        
    @abstractmethod
    def delete_project(self, name: str) -> None:
        """Delete a project."""
        pass

    # Family Methods
    @abstractmethod
    def create_family(self, name: str, project_name: str, description: str | None = None) -> FamilyDTO:
        """Create a new family in a project."""
        pass
        
    @abstractmethod
    def get_family(self, name: str, project_name: str) -> FamilyDTO | None:
        """Get a family by name and project."""
        pass
        
    @abstractmethod
    def list_families(self, project_name: str | None = None) -> list[FamilyDTO]:
        """List families, optionally filtered by project."""
        pass
        
    @abstractmethod
    def delete_family(self, name: str, project_name: str) -> None:
        """Delete a family."""
        pass

    @abstractmethod
    def add_experiment_to_family(self, family_name: str, project_name: str, experiment_name: str) -> None:
        """Add an experiment name to a family."""
        pass

    @abstractmethod
    def remove_experiment_from_family(self, family_name: str, project_name: str, experiment_name: str) -> None:
        """Remove an experiment name from a family."""
        pass
