
from typing import Literal, Union, Dict, List, Any
from pydantic import BaseModel, Field

class BaseDistribution(BaseModel):
    """Base class for parameter distributions."""
    type: str

class FloatDistribution(BaseDistribution):
    type: Literal["float"] = "float"
    low: float
    high: float
    log: bool = False
    step: float | None = None

class IntDistribution(BaseDistribution):
    type: Literal["int"] = "int"
    low: int
    high: int
    log: bool = False
    step: int = 1

class CategoricalDistribution(BaseDistribution):
    type: Literal["categorical"] = "categorical"
    choices: List[Union[str, int, float, bool]]

class SearchSpace(BaseModel):
    """
    Defines the search space for hyperparameter optimization.
    """
    parameters: Dict[str, Union[FloatDistribution, IntDistribution, CategoricalDistribution]]
