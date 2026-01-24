"""
Tests for KladML Package Validator.
"""

import pytest
import tempfile
import zipfile
import os
from pathlib import Path

from kladml.validator import (
    KladManifestParser,
    PackageValidator,
    ValidationResult,
    PackageType,
    validate_package,
)


# --- Test Fixtures ---

VALID_ARCHITECTURE_MANIFEST = """
klad_version: "1.0"
type: architecture
name: "TestModel"
version: "1.0.0"
description: "A test model"
author: "test"
model_type: classification

entry_point:
  module: "model"
  class: "TestModel"

input_contract:
  type: tabular
  
output_contract:
  type: prediction
  shape: [null, 1]
"""

VALID_PREPROCESSOR_MANIFEST = """
klad_version: "1.0"
type: preprocessor
name: "TestPreprocessor"
version: "1.0.0"

entry_point:
  module: "preprocessor"
  class: "TestPreprocessor"
"""

VALID_MODEL_PY = """
from kladml.models.base import BaseModel

class TestModel(BaseModel):
    def fit(self, X, y, **kwargs):
        self._is_fitted = True
    
    def predict(self, X, **kwargs):
        return X
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass
"""

VALID_PREPROCESSOR_PY = """
from kladml.data.preprocessor import BasePreprocessor

class TestPreprocessor(BasePreprocessor):
    def fit(self, dataset):
        self._is_fitted = True
    
    def transform(self, dataset):
        return dataset
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass
"""


def create_test_zip(tmpdir: str, manifest: str, entry_file: str, entry_content: str) -> str:
    """Create a test ZIP package."""
    # Create package directory
    pkg_dir = os.path.join(tmpdir, "test_package")
    os.makedirs(pkg_dir, exist_ok=True)
    
    # Write klad.yaml
    with open(os.path.join(pkg_dir, "klad.yaml"), "w") as f:
        f.write(manifest)
    
    # Write requirements.txt
    with open(os.path.join(pkg_dir, "requirements.txt"), "w") as f:
        f.write("numpy>=1.21.0\n")
    
    # Write entry point file
    with open(os.path.join(pkg_dir, entry_file), "w") as f:
        f.write(entry_content)
    
    # Create ZIP
    zip_path = os.path.join(tmpdir, "test_package.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for root, dirs, files in os.walk(pkg_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, pkg_dir)
                zf.write(file_path, arcname)
    
    return zip_path


# --- Manifest Parser Tests ---

class TestKladManifestParser:
    """Tests for KladManifestParser."""
    
    def test_parse_valid_architecture_manifest(self):
        """Test parsing a valid architecture manifest."""
        parser = KladManifestParser()
        manifest, errors = parser.parse(VALID_ARCHITECTURE_MANIFEST)
        
        assert len(errors) == 0
        assert manifest["name"] == "TestModel"
        assert manifest["type"] == "architecture"
        assert manifest["klad_version"] == "1.0"
    
    def test_parse_valid_preprocessor_manifest(self):
        """Test parsing a valid preprocessor manifest."""
        parser = KladManifestParser()
        manifest, errors = parser.parse(VALID_PREPROCESSOR_MANIFEST)
        
        assert len(errors) == 0
        assert manifest["name"] == "TestPreprocessor"
        assert manifest["type"] == "preprocessor"
    
    def test_parse_missing_required_field(self):
        """Test that missing required fields are detected."""
        parser = KladManifestParser()
        incomplete = """
klad_version: "1.0"
type: architecture
name: "Test"
# Missing: version, entry_point
"""
        manifest, errors = parser.parse(incomplete)
        
        assert len(errors) > 0
        assert any("entry_point" in e for e in errors)
    
    def test_parse_invalid_yaml(self):
        """Test handling of invalid YAML."""
        parser = KladManifestParser()
        invalid = """
klad_version: "1.0"
type: [not: valid: yaml
"""
        manifest, errors = parser.parse(invalid)
        
        assert manifest is None
        assert len(errors) > 0
        assert "Invalid YAML" in errors[0]
    
    def test_parse_invalid_type(self):
        """Test rejection of invalid type."""
        parser = KladManifestParser()
        invalid = """
klad_version: "1.0"
type: invalid_type
name: "Test"
version: "1.0.0"
entry_point:
  module: "test"
  class: "Test"
"""
        manifest, errors = parser.parse(invalid)
        
        assert len(errors) > 0
        assert any("Invalid type" in e for e in errors)
    
    def test_parse_unsupported_klad_version(self):
        """Test rejection of unsupported klad_version."""
        parser = KladManifestParser()
        invalid = """
klad_version: "99.0"
type: architecture
name: "Test"
version: "1.0.0"
entry_point:
  module: "test"
  class: "Test"
"""
        manifest, errors = parser.parse(invalid)
        
        assert len(errors) > 0
        assert any("Unsupported klad_version" in e for e in errors)


# --- Package Validator Tests ---

class TestPackageValidator:
    """Tests for PackageValidator."""
    
    def test_validate_valid_architecture_package(self):
        """Test validation of a valid architecture package."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = create_test_zip(
                tmpdir,
                VALID_ARCHITECTURE_MANIFEST,
                "model.py",
                VALID_MODEL_PY
            )
            
            result = validate_package(zip_path)
            
            assert result.valid, f"Errors: {[e.message for e in result.errors]}"
            assert result.package_type == PackageType.ARCHITECTURE
            assert result.manifest["name"] == "TestModel"
    
    def test_validate_valid_preprocessor_package(self):
        """Test validation of a valid preprocessor package."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = create_test_zip(
                tmpdir,
                VALID_PREPROCESSOR_MANIFEST,
                "preprocessor.py",
                VALID_PREPROCESSOR_PY
            )
            
            result = validate_package(zip_path)
            
            assert result.valid, f"Errors: {[e.message for e in result.errors]}"
            assert result.package_type == PackageType.PREPROCESSOR
    
    def test_validate_nonexistent_file(self):
        """Test handling of nonexistent file."""
        result = validate_package("/nonexistent/path.zip")
        
        assert not result.valid
        assert len(result.errors) > 0
        assert any("not found" in e.message.lower() for e in result.errors)
    
    def test_validate_not_a_zip(self):
        """Test handling of non-ZIP file."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(b"This is not a ZIP file")
            f.flush()
            
            result = validate_package(f.name)
            
            assert not result.valid
            assert any("valid ZIP" in e.message for e in result.errors)
            
            os.unlink(f.name)
    
    def test_validate_missing_klad_yaml(self):
        """Test detection of missing klad.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create ZIP without klad.yaml
            zip_path = os.path.join(tmpdir, "test.zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("requirements.txt", "numpy\n")
            
            result = validate_package(zip_path)
            
            assert not result.valid
            assert any("klad.yaml" in e.message for e in result.errors)
    
    def test_validate_missing_requirements(self):
        """Test detection of missing requirements.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create ZIP without requirements.txt
            zip_path = os.path.join(tmpdir, "test.zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("klad.yaml", VALID_ARCHITECTURE_MANIFEST)
                zf.writestr("model.py", VALID_MODEL_PY)
            
            result = validate_package(zip_path)
            
            assert not result.valid
            assert any("requirements.txt" in e.message for e in result.errors)
    
    def test_validate_missing_entry_point_module(self):
        """Test detection of missing entry point module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create ZIP without model.py
            zip_path = os.path.join(tmpdir, "test.zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("klad.yaml", VALID_ARCHITECTURE_MANIFEST)
                zf.writestr("requirements.txt", "numpy\n")
                # model.py is missing
            
            result = validate_package(zip_path)
            
            assert not result.valid
            assert any("Entry point module not found" in e.message for e in result.errors)
    
    def test_validate_missing_class(self):
        """Test detection of missing class in module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model.py without the expected class
            wrong_content = """
class WrongClassName:
    pass
"""
            zip_path = create_test_zip(
                tmpdir,
                VALID_ARCHITECTURE_MANIFEST,
                "model.py",
                wrong_content
            )
            
            result = validate_package(zip_path)
            
            # Should have error about missing class
            assert not result.valid or any("TestModel" in str(e) for e in result.errors + result.warnings)


class TestValidationResult:
    """Tests for ValidationResult."""
    
    def test_add_error_sets_valid_false(self):
        """Test that adding an error sets valid to False."""
        result = ValidationResult(valid=True)
        assert result.valid
        
        result.add_error("test", "Test error")
        
        assert not result.valid
        assert len(result.errors) == 1
    
    def test_add_warning_keeps_valid_true(self):
        """Test that adding a warning keeps valid True."""
        result = ValidationResult(valid=True)
        
        result.add_warning("test", "Test warning")
        
        assert result.valid
        assert len(result.warnings) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
