
import os
import logging
from typing import Dict, Any, Optional
import numpy as np

from kladml.models.timeseries.transformer.base import TransformerModel
from kladml.tasks import MLTask
from kladml.training.run_id import generate_run_id

logger = logging.getLogger(__name__)

class CanBusModel(TransformerModel):
    """
    Wrapper for CAN Bus Transformer Anomaly Detection.
    
    Implements:
    - Training Loop (Reconstruction Error)
    - Threshold Calculation
    - Anomaly Scoring
    """
    
    @property
    def ml_task(self) -> MLTask:
        # We can define a new task or use ANOMALY_DETECTION if available, 
        # otherwise Generic. Let's assume Generic or add Anomaly later.
        # For now, we reuse TIMESERIES_FORECASTING or define a custom property
        return MLTask.TIMESERIES_FORECASTING 
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.seq_len = self.config.get("seq_len", 120)
        self.num_features = self.config.get("num_features", 5)
        self.threshold = None
        self._scaler_stats = None

    def build_model(self):
        from kladml.models.timeseries.transformer.canbus.architecture import CanBusTransformer
        self.model = CanBusTransformer(
            num_features=self.num_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            e_layers=self.e_layers,
            dropout=self.dropout,
            activation=self.activation,
            seq_len=self.seq_len
        ).to(self.device)
        logger.info(f"Initialized CanBusTransformer on {self.device}")

    def train(self, X_train: Any, y_train: Any = None, X_val: Any = None, **kwargs) -> Dict[str, float]:
        import torch
        from torch.utils.data import DataLoader
        from kladml.models.timeseries.transformer.canbus.dataset import CanBusDataset
        
        # 1. Setup Run
        project_name = self.config.get("project_name", "default")
        experiment_name = self.config.get("experiment_name", "canbus")
        family_name = self.config.get("family_name", None)
        
        run_id = generate_run_id(project_name, experiment_name, family_name=family_name)
        self._init_standard_callbacks(run_id, project_name, experiment_name, family_name)
        
        # 2. Build or Load Model
        if self.model is None:
            self.build_model()
            
        # 3. Prepare Data
        # We assume X_train/X_val are paths to parquet files
        logger.info("Loading Training Data...")
        dataset_train = CanBusDataset(X_train, window_size=self.seq_len)
        self._scaler_stats = dataset_train.scaler_stats # Valid for saving later
        
        train_loader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        val_loader = None
        if X_val:
            logger.info("Loading Validation Data...")
            # Use same scaler stats!
            dataset_val = CanBusDataset(X_val, window_size=self.seq_len, scaler_stats=self._scaler_stats)
            val_loader = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=4)

        # 4. Optimization
        optimizer = self.configure_optimizers()
        criterion = torch.nn.MSELoss()
        
        # 5. Loop
        best_val_loss = float('inf')
        self.model.train()
        
        for epoch in range(self.config.get("epochs", 10)):
            self._callbacks_list.on_epoch_begin(epoch, {"epoch": epoch})
            
            # TRAIN
            train_losses = []
            self.model.train()
            for batch_idx, (x, target) in enumerate(train_loader):
                x = x.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                x_hat = self.model(x)
                loss = criterion(x_hat, target)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                
                if batch_idx % 500 == 0:
                    current_loss = loss.item()
                    print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {current_loss:.6f}", end='\r')
                    self._project_logger.log_metrics({"train_loss": current_loss, "step": batch_idx, "epoch": epoch})
            
            avg_train = np.mean(train_losses)
            metrics = {"train_loss": avg_train}
            
            # VAL
            if val_loader:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for x, target in val_loader:
                        x = x.to(self.device)
                        target = target.to(self.device)
                        x_hat = self.model(x)
                        loss = criterion(x_hat, target)
                        val_losses.append(loss.item())
                
                avg_val = np.mean(val_losses)
                metrics["val_loss"] = avg_val
                
                # Checkpoint
                is_best = avg_val < best_val_loss
                if is_best:
                    best_val_loss = avg_val
                    self._checkpoint_manager.save_checkpoint(
                        self.model, optimizer, epoch, metrics, is_best=True, scaler=self._scaler_stats
                    )
            
            self._callbacks_list.on_epoch_end(epoch, metrics)
            logger.info(f"Epoch {epoch} Results: {metrics}")
            
        return metrics

    
    def evaluate(self, X_test: Any, y_test: Any = None, **kwargs) -> Dict[str, float]:
        """Evaluate model on test set (calculate reconstruction error)."""
        import torch
        from torch.utils.data import DataLoader
        from kladml.models.timeseries.transformer.canbus.dataset import CanBusDataset
        
        if self.model is None:
             raise ValueError("Model not initialized")
             
        self.model.eval()
        criterion = torch.nn.MSELoss()
        losses = []
        
        # Load Data
        dataset = CanBusDataset(X_test, window_size=self.seq_len, scaler_stats=self._scaler_stats)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        with torch.no_grad():
            for x, target in loader:
                x = x.to(self.device)
                target = target.to(self.device)
                x_hat = self.model(x)
                loss = criterion(x_hat, target)
                losses.append(loss.item())
                
        metrics = {"test_loss": np.mean(losses)}
        logger.info(f"Evaluation Results: {metrics}")
        return metrics

    def predict(self, X: Any, threshold: float = 1.0, sensitivity: float = 1.0) -> Dict[str, Any]:
        """
        Compute Anomaly Scores (0 to 1).
        
        Logic:
        - Calculates reconstruction error (MSE).
        - Normalizes to [0, 1] using a Sigmoid function centered at `threshold`.
        - Score 0.5 means Error == Threshold.
        
        Args:
            X: Input data (Tensor or Numpy) [Batch, Seq, Feat]
            threshold: MSE value considered as the boundary (Score=0.5).
            sensitivity: Scaling factor. Higher = sharper transition.
            
        Returns:
            Dict containing:
            - scores: np.array [Batch] (0.0 to 1.0)
            - raw_errors: np.array [Batch] (MSE)
            - reconstructions: np.array [Batch, Seq, Feat]
        """
        if self.model is None:
             raise ValueError("Model not trained")
             
        import torch
        self.model.eval()
        
        # Prepare input
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        with torch.no_grad():
            x_hat = self.model(X)
            
            # MSE per sample
            loss = torch.mean((x_hat - X) ** 2, dim=[1, 2]) # [Batch]
            raw_errors = loss.cpu().numpy()
            
            # Sigmoid Scoring
            # Score = 1 / (1 + exp( - (error - threshold) * scaler ))
            # We want: error=threshold -> 0.5
            # We want: error=0 -> close to 0
            # Heuristic scaling: sensitivity / threshold
            scale = sensitivity * (10.0 / threshold) if threshold > 0 else 1.0
            scores = 1.0 / (1.0 + np.exp( - (raw_errors - threshold) * scale ))
            
        return {
            "scores": scores,
            "raw_errors": raw_errors,
            "reconstructions": x_hat.cpu().numpy()
        }
