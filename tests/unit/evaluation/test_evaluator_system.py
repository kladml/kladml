
import pytest
import torch
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from kladml.tasks import MLTask
from kladml.evaluation.registry import EvaluatorRegistry
from kladml.evaluation.classification.evaluator import ClassificationEvaluator
from kladml.evaluation.regression.evaluator import RegressionEvaluator

# ... (Previous fixtures) ...
# Mock data
@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run_eval"
    d.mkdir()
    d.joinpath("plots").mkdir(parents=True, exist_ok=True)
    return d

@pytest.fixture
def dummy_artifacts(tmp_path):
    """Create dummy model and data files."""
    model = torch.nn.Linear(2, 2)
    model_path = tmp_path / "model.pt"
    torch.save(model, model_path)
    
    data = (torch.randn(4, 2), torch.randint(0, 2, (4,)))
    data_path = tmp_path / "data.pt"
    torch.save(data, data_path)
    
    return model_path, data_path, model, data

def test_registry_discovery():
    assert EvaluatorRegistry.get_evaluator(MLTask.CLASSIFICATION) == ClassificationEvaluator
    assert EvaluatorRegistry.get_evaluator("classification") == ClassificationEvaluator
    with pytest.raises(ValueError):
        EvaluatorRegistry.get_evaluator("non_existent_task")

def test_load_and_inference_classification(run_dir, dummy_artifacts):
    model_path, data_path, _, _ = dummy_artifacts
    evaluator = ClassificationEvaluator(
        run_dir=run_dir, model_path=model_path, data_path=data_path, config={"num_classes": 2}
    )
    model = evaluator.load_model()
    data = evaluator.load_data()
    preds, targets = evaluator.inference(model, data)
    assert preds.shape == (4, 2)
    
    evaluator.model_path = Path("non_existent.pt")
    assert evaluator.load_model() is None

def test_load_and_inference_regression(run_dir, dummy_artifacts):
    model_path, data_path, _, _ = dummy_artifacts
    evaluator = RegressionEvaluator(
        run_dir=run_dir, model_path=model_path, data_path=data_path
    )
    model = evaluator.load_model()
    data = evaluator.load_data()
    preds, targets = evaluator.inference(model, data)
    assert preds.shape == (4, 2)
    
    with pytest.raises(NotImplementedError):
        evaluator.inference(model, "invalid_data_format")

def test_base_error_handling(run_dir):
    evaluator = ClassificationEvaluator(
        run_dir=run_dir, model_path=Path("dummy"), data_path=Path("dummy")
    )
    evaluator.load_model = MagicMock(side_effect=Exception("Boom"))
    with pytest.raises(Exception, match="Boom"):
        evaluator.run()

def test_full_lifecycle(run_dir):
    evaluator = ClassificationEvaluator(
        run_dir=run_dir, model_path=Path("dummy.pt"), data_path=Path("dummy.pt"), config={"num_classes": 2}
    )
    evaluator.load_model = MagicMock(return_value=torch.nn.Linear(2,2))
    evaluator.load_data = MagicMock(return_value="mock_data")
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.9, 0.1], [0.2, 0.8]])
    targets = torch.tensor([1, 0, 0, 1])
    evaluator.inference = MagicMock(return_value=(preds, targets))
    
    metrics = evaluator.run()
    assert "accuracy" in metrics
    assert (run_dir / "evaluation_report.md").exists()
    assert (run_dir / "plots" / "confusion_matrix.png").exists()

def test_lifecycle_regression(run_dir):
    """Test full cycle for regression."""
    evaluator = RegressionEvaluator(
        run_dir=run_dir, model_path=Path("dummy.pt"), data_path=Path("dummy.pt")
    )
    evaluator.load_model = MagicMock()
    evaluator.load_data = MagicMock()
    # Preds (N, 1) or (N,), Targets (N,)
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.1, 1.9, 3.1])
    evaluator.inference = MagicMock(return_value=(preds, targets))
    
    metrics = evaluator.run()
    
    assert "mae" in metrics
    assert (run_dir / "plots" / "residuals.png").exists()
    assert (run_dir / "plots" / "pred_vs_actual.png").exists()
