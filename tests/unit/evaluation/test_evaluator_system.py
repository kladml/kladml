
import pytest
import torch
import shutil
from pathlib import Path
from kladml.tasks import MLTask
from kladml.evaluation.registry import EvaluatorRegistry
from kladml.evaluation.classification.evaluator import ClassificationEvaluator
from kladml.evaluation.regression.evaluator import RegressionEvaluator

# Mock data
@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run_eval"
    d.mkdir()
    (d / "plots").mkdir() # Evaluator usually creates this? BaseEvaluator: self.plots_dir.mkdir(parents=True, exist_ok=True)
    return d

def test_registry_discovery():
    """Test registry returns correct classes."""
    assert EvaluatorRegistry.get_evaluator(MLTask.CLASSIFICATION) == ClassificationEvaluator
    # Fuzzy string matching
    assert EvaluatorRegistry.get_evaluator("classification") == ClassificationEvaluator
    
    with pytest.raises(ValueError):
        EvaluatorRegistry.get_evaluator("non_existent_task")

def test_classification_metrics(run_dir):
    """Test classification metrics computation."""
    evaluator = ClassificationEvaluator(
        run_dir=run_dir,
        model_path=Path("dummy"),
        data_path=Path("dummy"),
        config={"num_classes": 2}
    )
    
    # Perfect predictions
    preds = torch.tensor([0, 1, 0, 1])
    targets = torch.tensor([0, 1, 0, 1])
    
    metrics = evaluator.compute_metrics(preds, targets)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    
    # Mixed
    preds = torch.tensor([0, 0, 0, 1])
    targets = torch.tensor([0, 1, 0, 1]) # Acc: 3/4 = 0.75
    
    metrics = evaluator.compute_metrics(preds, targets)
    assert metrics["accuracy"] == 0.75

def test_regression_metrics(run_dir):
    """Test regression metrics computation."""
    evaluator = RegressionEvaluator(
        run_dir=run_dir,
        model_path=Path("dummy"),
        data_path=Path("dummy")
    )
    
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 2.0, 3.0])
    
    metrics = evaluator.compute_metrics(preds, targets)
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    
    # Error case
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([2.0, 3.0, 4.0]) # Diff 1.0
    
    metrics = evaluator.compute_metrics(preds, targets)
    assert metrics["mae"] == 1.0
    assert metrics["mse"] == 1.0

def test_plot_generation(run_dir):
    """Test that plots are saved."""
    evaluator = ClassificationEvaluator(
        run_dir=run_dir,
        model_path=Path("dummy"),
        data_path=Path("dummy")
    )
    
    preds = torch.tensor([0, 1, 0, 1])
    targets = torch.tensor([0, 1, 0, 1])
    
    evaluator.save_plots(preds, targets)
    
    assert (run_dir / "plots" / "confusion_matrix.png").exists()

