"""
Example: Simple Training Script with KladML

This demonstrates how to use KladML SDK for standalone training.
"""

from kladml import (
    TimeSeriesModel,
    LocalStorage,
    YamlConfig,
    ConsolePublisher,
    LocalTracker,
    ExperimentRunner,
    MLTask,
)


class SimpleForecaster(TimeSeriesModel):
    """A minimal example forecasting model."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
    
    def train(self, X_train, y_train=None, **kwargs):
        """Simulate training."""
        import time
        
        epochs = self.config.get("epochs", 5)
        print(f"Training for {epochs} epochs...")
        
        # Simulate training loop
        for epoch in range(epochs):
            loss = 1.0 / (epoch + 1)  # Decreasing loss
            print(f"  Epoch {epoch+1}/{epochs} - loss: {loss:.4f}")
            time.sleep(0.1)  # Simulate work
        
        self._is_trained = True
        return {"final_loss": loss, "epochs": epochs}
    
    def predict(self, X, **kwargs):
        """Generate dummy predictions."""
        import numpy as np
        return np.zeros(len(X))
    
    def evaluate(self, X_test, y_test=None, **kwargs):
        """Simulate evaluation."""
        return {"mae": 0.15, "mse": 0.05}
    
    def save(self, path):
        """Save model state."""
        import json
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(f"{path}/config.json", "w") as f:
            json.dump(self.config, f)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model state."""
        import json
        with open(f"{path}/config.json") as f:
            self.config = json.load(f)


def main():
    print("=" * 50)
    print("KladML SDK - Standalone Training Example")
    print("=" * 50)
    
    # Initialize backends (all local, no server needed)
    config = YamlConfig()
    storage = LocalStorage(config.artifacts_dir)
    publisher = ConsolePublisher()
    
    # Create experiment runner with local backends
    runner = ExperimentRunner(
        config=config,
        storage=storage,
        publisher=publisher,
        # tracker=LocalTracker(),  # Optional: enable MLflow tracking
    )
    
    # Run experiment
    import numpy as np
    dummy_data = np.random.randn(100, 10)  # 100 samples, 10 features
    
    result = runner.run(
        model_class=SimpleForecaster,
        train_data=dummy_data,
        val_data=dummy_data[:20],
        experiment_name="example-experiment",
        run_name="test-run-1",
        model_config={"epochs": 3, "window_size": 10},
    )
    
    print("\n" + "=" * 50)
    print("Results:")
    print(f"  Run ID: {result['run_id']}")
    print(f"  Status: {result['status']}")
    print(f"  Train Metrics: {result['train_metrics']}")
    print(f"  Eval Metrics: {result['eval_metrics']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
