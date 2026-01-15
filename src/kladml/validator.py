"""
KladML Package Validator

Validates uploaded ZIP packages against the KladML packaging standard.
Parses klad.yaml and verifies interface compliance.
"""

import yaml
import zipfile
import tempfile
import importlib.util
import inspect
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class PackageType(Enum):
    """Type of package being validated."""
    ARCHITECTURE = "architecture"
    PREPROCESSOR = "preprocessor"


@dataclass
class ValidationError:
    """Represents a validation error."""
    field: str
    message: str
    severity: str = "error"  # error, warning


@dataclass 
class ValidationResult:
    """Result of package validation."""
    valid: bool
    package_type: Optional[PackageType] = None
    manifest: Optional[Dict[str, Any]] = None
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    
    def add_error(self, field: str, message: str):
        self.errors.append(ValidationError(field, message, "error"))
        self.valid = False
    
    def add_warning(self, field: str, message: str):
        self.warnings.append(ValidationError(field, message, "warning"))


class KladManifestParser:
    """Parses and validates klad.yaml manifest files."""
    
    REQUIRED_FIELDS = ["klad_version", "type", "name", "version", "entry_point"]
    VALID_TYPES = ["architecture", "preprocessor"]
    SUPPORTED_KLAD_VERSIONS = ["1.0"]
    
    def parse(self, yaml_content: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Parse klad.yaml content.
        
        Returns:
            Tuple of (parsed_dict, list_of_errors)
        """
        errors = []
        
        try:
            manifest = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            return None, [f"Invalid YAML syntax: {e}"]
        
        if not isinstance(manifest, dict):
            return None, ["klad.yaml must be a YAML mapping"]
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in manifest:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return manifest, errors
        
        # Validate klad_version
        if manifest.get("klad_version") not in self.SUPPORTED_KLAD_VERSIONS:
            errors.append(f"Unsupported klad_version: {manifest.get('klad_version')}. Supported: {self.SUPPORTED_KLAD_VERSIONS}")
        
        # Validate type
        if manifest.get("type") not in self.VALID_TYPES:
            errors.append(f"Invalid type: {manifest.get('type')}. Must be one of: {self.VALID_TYPES}")
        
        # Validate entry_point structure
        entry_point = manifest.get("entry_point", {})
        if not isinstance(entry_point, dict):
            errors.append("entry_point must be a mapping with 'module' and 'class' keys")
        elif "module" not in entry_point or "class" not in entry_point:
            errors.append("entry_point must contain both 'module' and 'class' keys")
        
        return manifest, errors


class PackageValidator:
    """Validates KladML packages (ZIP files)."""
    
    REQUIRED_FILES = ["klad.yaml", "requirements.txt"]
    
    ARCHITECTURE_REQUIRED_METHODS = ["fit", "predict", "save", "load"]
    PREPROCESSOR_REQUIRED_METHODS = ["fit", "transform", "save", "load"]
    
    def __init__(self):
        self.manifest_parser = KladManifestParser()
    
    def validate_zip(self, zip_path: str) -> ValidationResult:
        """
        Validate a ZIP package.
        
        Args:
            zip_path: Path to the ZIP file
        
        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        result = ValidationResult(valid=True)
        
        # Check file exists
        if not os.path.exists(zip_path):
            result.add_error("file", f"File not found: {zip_path}")
            return result
        
        # Check it's a valid ZIP
        if not zipfile.is_zipfile(zip_path):
            result.add_error("file", "Not a valid ZIP file")
            return result
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Extract ZIP
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(tmpdir)
                
                # Find package root (handle nested directories)
                package_root = self._find_package_root(tmpdir)
                if not package_root:
                    result.add_error("structure", "Could not find klad.yaml in archive")
                    return result
                
                # Validate structure
                self._validate_structure(package_root, result)
                if not result.valid:
                    return result
                
                # Parse and validate manifest
                manifest_path = os.path.join(package_root, "klad.yaml")
                with open(manifest_path, 'r') as f:
                    manifest_content = f.read()
                
                manifest, parse_errors = self.manifest_parser.parse(manifest_content)
                for error in parse_errors:
                    result.add_error("klad.yaml", error)
                
                if not result.valid:
                    return result
                
                result.manifest = manifest
                result.package_type = PackageType(manifest["type"])
                
                # Validate entry point exists
                self._validate_entry_point(package_root, manifest, result)
                
                # Validate interface (if entry point exists)
                if result.valid:
                    self._validate_interface(package_root, manifest, result)
                
            except Exception as e:
                result.add_error("validation", f"Validation failed: {str(e)}")
        
        return result
    
    def _find_package_root(self, tmpdir: str) -> Optional[str]:
        """Find the directory containing klad.yaml."""
        # Check root level
        if os.path.exists(os.path.join(tmpdir, "klad.yaml")):
            return tmpdir
        
        # Check one level deep
        for item in os.listdir(tmpdir):
            item_path = os.path.join(tmpdir, item)
            if os.path.isdir(item_path):
                if os.path.exists(os.path.join(item_path, "klad.yaml")):
                    return item_path
        
        return None
    
    def _validate_structure(self, package_root: str, result: ValidationResult):
        """Validate required files exist."""
        for required_file in self.REQUIRED_FILES:
            file_path = os.path.join(package_root, required_file)
            if not os.path.exists(file_path):
                result.add_error("structure", f"Missing required file: {required_file}")
    
    def _validate_entry_point(self, package_root: str, manifest: Dict, result: ValidationResult):
        """Validate that the entry point module exists."""
        entry_point = manifest.get("entry_point", {})
        module_name = entry_point.get("module", "")
        
        # Check for .py file
        module_file = os.path.join(package_root, f"{module_name}.py")
        if not os.path.exists(module_file):
            result.add_error("entry_point", f"Entry point module not found: {module_name}.py")
    
    def _validate_interface(self, package_root: str, manifest: Dict, result: ValidationResult):
        """Validate that the class implements required interface."""
        entry_point = manifest.get("entry_point", {})
        module_name = entry_point.get("module", "")
        class_name = entry_point.get("class", "")
        
        module_file = os.path.join(package_root, f"{module_name}.py")
        
        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(module_name, module_file)
            if spec is None or spec.loader is None:
                result.add_warning("interface", "Could not load module for interface validation")
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get class
            if not hasattr(module, class_name):
                result.add_error("interface", f"Class '{class_name}' not found in module '{module_name}'")
                return
            
            cls = getattr(module, class_name)
            
            # Check required methods
            package_type = PackageType(manifest["type"])
            if package_type == PackageType.ARCHITECTURE:
                required_methods = self.ARCHITECTURE_REQUIRED_METHODS
            else:
                required_methods = self.PREPROCESSOR_REQUIRED_METHODS
            
            for method in required_methods:
                if not hasattr(cls, method) or not callable(getattr(cls, method)):
                    result.add_error("interface", f"Missing required method: {method}")
        
        except Exception as e:
            result.add_warning("interface", f"Could not validate interface: {str(e)}")


def validate_package(zip_path: str) -> ValidationResult:
    """
    Convenience function to validate a package.
    
    Args:
        zip_path: Path to the ZIP file
    
    Returns:
        ValidationResult
    """
    validator = PackageValidator()
    return validator.validate_zip(zip_path)
