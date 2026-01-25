
"""
Evaluation Module for KladML.

Provides base classes and concrete evaluators.
"""

from .base import BaseEvaluator
from .classification.evaluator import ClassificationEvaluator
from .regression.evaluator import RegressionEvaluator
from .registry import EvaluatorRegistry

__all__ = [
    "BaseEvaluator", 
    "ClassificationEvaluator",
    "RegressionEvaluator",
    "EvaluatorRegistry"
]
