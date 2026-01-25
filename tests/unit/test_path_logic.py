
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from kladml.utils.paths import resolve_dataset_path, ensure_data_structure, DB_DIR
from kladml.db.session import get_db_path

def test_resolve_paths(tmp_path):
    # Mock CWD to tmp_path
    with patch("kladml.utils.paths.Path.cwd", return_value=tmp_path):
        # 1. Resolve relative default
        p = resolve_dataset_path("my_data.pkl")
        expected = tmp_path / "data/datasets/my_data.pkl"
        assert p == expected
        
        # 2. Resolve absolute
        abs_p = Path("/tmp/foo.pkl")
        p = resolve_dataset_path(str(abs_p))
        assert p == abs_p

def test_ensure_structure(tmp_path):
    with patch("kladml.utils.paths.Path.cwd", return_value=tmp_path):
        ensure_data_structure()
        
        assert (tmp_path / "data/datasets").exists()
        assert (tmp_path / "data/projects").exists()
        assert (tmp_path / "data/db").exists()
        assert (tmp_path / "data/preprocessors").exists()

def test_db_path_default(tmp_path):
    with patch("kladml.utils.paths.Path.cwd", return_value=tmp_path):
        # Reset settings mock if needed, but assuming default env
        with patch("kladml.db.session.settings") as mock_settings:
            mock_settings.database_url = "sqlite:///kladml.db"
            
            p = get_db_path()
            # It should be in data/db/kladml.db now
            expected = tmp_path / "data/db/kladml.db"
            assert p == expected

def test_db_path_explicit(tmp_path):
    with patch("kladml.utils.paths.Path.cwd", return_value=tmp_path):
         with patch("kladml.db.session.settings") as mock_settings:
            mock_settings.database_url = "sqlite:///custom.db"
            p = get_db_path()
            expected = tmp_path / "data/db/custom.db"
            assert p == expected
