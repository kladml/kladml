
import optuna
import os
from typing import Callable, Any, Dict, Optional
from dataclasses import dataclass
from .search_space import SearchSpace, FloatDistribution, IntDistribution, CategoricalDistribution

@dataclass
class TunerConfig:
    study_name: str
    n_trials: int = 20
    direction: str = "minimize"
    storage: Optional[str] = None # Defaults to settings.database_url if not provided

class KladMLTuner:
    """
    Wrapper around Optuna Study to simplify optimization.
    """
    def __init__(self, config: TunerConfig):
        self.config = config
        
        # Default to global settings DB if not provided
        if not config.storage:
            from kladml.config.settings import settings
            # Optuna needs standard sqlite:/// path. settings.database_url is usually correct.
            # But we might need to strip async prefixes if using asyncpg? No, we use sqlite.
            storage_url = settings.database_url
        else:
            storage_url = config.storage

        # Initialize Study
        self.study = optuna.create_study(
            study_name=config.study_name,
            direction=config.direction,
            storage=storage_url,
            load_if_exists=True
        )
        
    def optimize(self, objective: Callable[[Dict[str, Any]], float], space: SearchSpace) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Args:
            objective: Function taking `params` dict and returning float score.
            space: SearchSpace definition.
            
        Returns:
            Best parameters found.
        """
        
        def optuna_objective(trial: optuna.Trial) -> float:
            # 1. Provide Params
            params = {}
            for name, dist in space.parameters.items():
                if isinstance(dist, FloatDistribution):
                    params[name] = trial.suggest_float(name, dist.low, dist.high, log=dist.log, step=dist.step)
                elif isinstance(dist, IntDistribution):
                    params[name] = trial.suggest_int(name, dist.low, dist.high, log=dist.log, step=dist.step)
                elif isinstance(dist, CategoricalDistribution):
                    params[name] = trial.suggest_categorical(name, dist.choices)
            
            # 2. Call User Objective
            import inspect
            sig = inspect.signature(objective)
            if len(sig.parameters) >= 2:
                score = objective(params, trial)
            else:
                score = objective(params)
            
            # 3. Handle Pruning
            # If objective raises TrialPruned, Optuna catches it.
            
            return score
            
        self.study.optimize(optuna_objective, n_trials=self.config.n_trials)
        
        return self.study.best_params
