
"""
Evaluator Registry.

Maps ML Tasks to their default Evaluators.
"""

from typing import Type
from kladml.tasks import MLTask
from kladml.evaluation.base import BaseEvaluator
from kladml.evaluation.classification.evaluator import ClassificationEvaluator
from kladml.evaluation.regression.evaluator import RegressionEvaluator
from kladml.evaluation.timeseries.evaluator import TimeSeriesEvaluator

class EvaluatorRegistry:
    """Registry for discovering Evaluators."""
    
    _REGISTRY = {
        MLTask.CLASSIFICATION: ClassificationEvaluator,
        MLTask.REGRESSION: RegressionEvaluator,
        MLTask.TIMESERIES_FORECASTING: TimeSeriesEvaluator,
    }
    
    @classmethod
    def get_evaluator(cls, task: MLTask | str) -> Type[BaseEvaluator]:
        """
        Get the default evaluator class for a given task.
        
        Args:
            task: MLTask enum or string (e.g., "classification").
            
        Returns:
            Evaluator class.
            
        Raises:
            ValueError: If no evaluator is found for the task.
        """
        if isinstance(task, str):
            try:
                task = MLTask(task.lower())
            except ValueError:
                # Try to fuzzy match keys if exact match fails
                pass

        if task in cls._REGISTRY:
            return cls._REGISTRY[task]
            
        raise ValueError(f"No default evaluator for task: {task}")

    @classmethod
    def register(cls, task: MLTask, evaluator_cls: Type[BaseEvaluator]):
        """Register a new evaluator for a task."""
        cls._REGISTRY[task] = evaluator_cls
