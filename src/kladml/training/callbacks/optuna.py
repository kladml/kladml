
from typing import Dict, Any
import optuna
from ...training.callbacks import Callback

class OptunaPruningCallback(Callback):
    """
    Reports validation metrics to Optuna and checks for pruning.
    If pruned, raises optuna.TrialPruned().
    """
    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Any, val_metrics: Dict[str, float]):
        """Called at end of validation loop."""
        current_score = val_metrics.get(self.monitor)
        
        if current_score is None:
            return

        # Report to Optuna
        step = trainer.current_epoch
        self.trial.report(current_score, step=step)

        # Check for pruning
        if self.trial.should_prune():
            message = f"Trial pruned at epoch {step} with {self.monitor}={current_score}"
            # Trainer needs to handle TrialPruned or we let it bubble up?
            # If we raise TrialPruned here, Trainer crashes.
            # KladML Trainer loop typically catches Exceptions?
            # We should probably raise a specific StopTraining exception designated for pruning?
            # But Optuna expects TrialPruned.
            # If Trainer doesn't catch it, it bubbles to Tuner.optimize which handles it.
            # BUT Trainer might have cleanup (save_checkpoint, etc).
            # We assume bubbling up is fine for now as it stops the run universally.
            raise optuna.TrialPruned(message)
