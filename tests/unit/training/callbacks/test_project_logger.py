
import pytest
import json
from kladml.training.callbacks.project_logger import ProjectLogger

class TestProjectLogger:
    """Tests for ProjectLogger callback."""
    
    @pytest.fixture
    def logger_instance(self, tmp_path):
        """Create a ProjectLogger instance with a temporary directory."""
        return ProjectLogger(
            project_name="test_proj",
            experiment_name="test_exp",
            run_id="run_123",
            projects_dir=str(tmp_path),
            log_format="jsonl",
            console_output=False
        )

    def test_directory_structure(self, logger_instance, tmp_path):
        """Test directory creation."""
        expected_dir = tmp_path / "test_proj" / "test_exp" / "run_123"
        assert expected_dir.exists()
        assert logger_instance.log_dir == expected_dir
        assert (expected_dir / "training.jsonl").exists()

    def test_log_jsonl_format(self, logger_instance):
        """Test logging in JSONL format."""
        logger_instance.info("Test message", {"acc": 0.99})
        logger_instance._close_file() # Ensure flush
        
        with open(logger_instance.log_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            
            assert entry["level"] == "info"
            assert entry["message"] == "Test message"
            assert entry["data"]["acc"] == 0.99
            assert "timestamp" in entry

    def test_log_text_format(self, tmp_path):
        """Test logging in plain text format."""
        logger = ProjectLogger(
            project_name="test_proj",
            experiment_name="test_exp_txt",
            run_id="run_456",
            projects_dir=str(tmp_path),
            log_format="text",
            console_output=False
        )
        
        logger.warning("Warning message", {"loss": 0.5})
        logger._close_file()
        
        expected_file = tmp_path / "test_proj" / "test_exp_txt" / "run_456" / "training.log"
        assert expected_file.exists()
        
        with open(expected_file) as f:
            content = f.read()
            assert "[WARNING] Warning message" in content
            assert "{'loss': 0.5}" in content

    def test_callback_hooks(self, logger_instance):
        """Test callback integration hooks."""
        # Train start
        logger_instance.on_train_begin({"lr": 0.001})
        
        # Epoch start/end
        logger_instance.on_epoch_begin(1)
        logger_instance.on_epoch_end(1, {"val_loss": 0.2})
        
        # Train end
        logger_instance.on_train_end()
        
        # Verify file content
        with open(logger_instance.log_file) as f:
            lines = [json.loads(line) for line in f.readlines()]
            
        messages = [l["message"] for l in lines]
        assert "Training started" in messages
        assert "Epoch 1 started" in messages
        assert "Epoch 1 completed" in messages
        assert "Training completed" in messages
        
        # Verify data passed correctly
        assert lines[0]["data"]["lr"] == 0.001
        assert lines[2]["data"]["val_loss"] == 0.2

    def test_batch_logging_frequency(self, logger_instance):
        """Test that batch logging is throttled."""
        # Batch 0 (Should log)
        logger_instance.on_batch_end(0)
        
        # Batch 1 (Should NOT log)
        logger_instance.on_batch_end(1)
        
        # Batch 100 (Should log)
        logger_instance.on_batch_end(100)
        
        logger_instance._close_file()
        
        with open(logger_instance.log_file) as f:
            lines = [json.loads(line) for line in f.readlines()]
            
        assert len(lines) == 2
        assert lines[0]["message"] == "Batch 0"
        assert lines[1]["message"] == "Batch 100"
