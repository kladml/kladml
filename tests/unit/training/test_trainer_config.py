
import pytest
from pydantic import ValidationError
from kladml.training.trainer import UniversalTrainer
from kladml.config.schema import TrainingConfig

class TestTrainerConfig:
    """Tests for UniversalTrainer configuration logic."""
    
    def test_init_with_explicit_args(self):
        """Test backward compatibility with explicit arguments."""
        trainer = UniversalTrainer(max_epochs=5, accelerator="cpu")
        assert trainer.max_epochs == 5
        assert trainer.config.accelerator == "cpu"
        assert isinstance(trainer.config, TrainingConfig)

    def test_init_with_config_object(self):
        """Test initialization with TrainingConfig object."""
        config = TrainingConfig(max_epochs=20, accelerator="gpu")
        trainer = UniversalTrainer(config=config)
        assert trainer.max_epochs == 20
        assert trainer.config.accelerator == "gpu"
        assert trainer.config is config

    def test_init_with_dict(self):
        """Test initialization with dictionary (auto-conversion)."""
        config_dict = {"max_epochs": 15, "accelerator": "mps"}
        trainer = UniversalTrainer(config=config_dict)
        assert trainer.max_epochs == 15
        assert trainer.config.accelerator == "mps"
        assert isinstance(trainer.config, TrainingConfig)

    def test_validation_error(self):
        """Test that invalid config raises ValidationError."""
        # max_epochs must be >= 1
        with pytest.raises(ValidationError):
            UniversalTrainer(config={"max_epochs": 0})
            
        # accelerator must be valid literal
        with pytest.raises(ValidationError):
            UniversalTrainer(config={"accelerator": "invalid_device"})

    def test_extra_fields_forbidden(self):
        """Test that unknown fields are forbidden."""
        with pytest.raises(ValidationError):
            UniversalTrainer(config={"max_epochs": 10, "unknown_field": 123})
