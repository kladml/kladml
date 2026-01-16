"""
Base Evaluator for KladML.

Abstract base class defining the evaluation contract.
Uses Template Method pattern for the evaluation pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import json


class BaseEvaluator(ABC):
    """
    Abstract base class for model evaluation.
    
    Provides the Template Method pattern for evaluation:
    1. load_model()
    2. load_data()
    3. inference()
    4. compute_metrics()
    5. save_plots()
    6. generate_report()
    
    Subclasses must implement abstract methods.
    """
    
    def __init__(
        self, 
        run_dir: Path, 
        model_path: Path, 
        data_path: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            run_dir: Directory where evaluation outputs will be saved.
            model_path: Path to the model checkpoint.
            data_path: Path to the evaluation dataset.
            config: Optional configuration dictionary.
        """
        self.run_dir = Path(run_dir)
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.config = config or {}
        
        # Output directories
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.metrics: Dict[str, float] = {}
        self.predictions = None
        self.targets = None
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        
        # Initialize logger
        self._logger = self._init_logger()
    
    def _init_logger(self) -> logging.Logger:
        """Initialize file logger for evaluation."""
        logger = logging.getLogger(f"evaluator.{self.__class__.__name__}")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers = []
        
        # File handler
        log_file = self.run_dir / "evaluation.log"
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        
        # Format
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    @abstractmethod
    def load_model(self) -> Any:
        """
        Load the model from checkpoint.
        
        Returns:
            Loaded model ready for inference.
        """
        pass
    
    @abstractmethod
    def load_data(self) -> Any:
        """
        Load evaluation dataset.
        
        Returns:
            Loaded dataset or dataloader.
        """
        pass
    
    @abstractmethod
    def inference(self, model: Any, data: Any) -> Tuple[Any, Any]:
        """
        Run inference on the data.
        
        Args:
            model: The loaded model.
            data: The loaded data.
            
        Returns:
            Tuple of (predictions, targets).
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, predictions: Any, targets: Any) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            
        Returns:
            Dictionary of metric names to values.
        """
        pass
    
    @abstractmethod
    def save_plots(self, predictions: Any, targets: Any) -> None:
        """
        Generate and save evaluation plots.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
        """
        pass
    
    @abstractmethod
    def generate_report(self) -> str:
        """
        Generate Markdown report content.
        
        Returns:
            Markdown string with the evaluation report.
        """
        pass
    
    def _save_report(self, content: str) -> Path:
        """Save the Markdown report to disk."""
        report_path = self.run_dir / "evaluation_report.md"
        report_path.write_text(content, encoding="utf-8")
        self._logger.info(f"Saved report: {report_path}")
        return report_path
    
    def _save_metrics_json(self) -> Path:
        """Save metrics as JSON for programmatic access."""
        metrics_path = self.run_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "metrics": self.metrics,
                "timestamp": datetime.now().isoformat(),
                "model_path": str(self.model_path),
                "data_path": str(self.data_path),
            }, f, indent=2)
        self._logger.info(f"Saved metrics: {metrics_path}")
        return metrics_path
    
    def run(self) -> Dict[str, float]:
        """
        Template Method: Execute the full evaluation pipeline.
        
        Returns:
            Dictionary of computed metrics.
        """
        self._start_time = datetime.now()
        self._logger.info("=" * 60)
        self._logger.info("EVALUATION STARTED")
        self._logger.info(f"Model: {self.model_path}")
        self._logger.info(f"Data: {self.data_path}")
        self._logger.info(f"Output: {self.run_dir}")
        self._logger.info("=" * 60)
        
        try:
            # Step 1: Load Model
            self._logger.info("Loading model...")
            model = self.load_model()
            
            # Step 2: Load Data
            self._logger.info("Loading data...")
            data = self.load_data()
            
            # Step 3: Inference
            self._logger.info("Running inference...")
            self.predictions, self.targets = self.inference(model, data)
            
            # Step 4: Compute Metrics
            self._logger.info("Computing metrics...")
            self.metrics = self.compute_metrics(self.predictions, self.targets)
            
            for name, value in self.metrics.items():
                self._logger.info(f"  {name}: {value:.4f}")
            
            # Step 5: Save Plots
            self._logger.info("Generating plots...")
            self.save_plots(self.predictions, self.targets)
            
            # Step 6: Generate Report
            self._logger.info("Generating report...")
            report = self.generate_report()
            self._save_report(report)
            
            # Save metrics JSON
            self._save_metrics_json()
            
        except Exception as e:
            self._logger.error(f"Evaluation failed: {e}")
            raise
        
        finally:
            self._end_time = datetime.now()
            duration = (self._end_time - self._start_time).total_seconds()
            self._logger.info("=" * 60)
            self._logger.info(f"EVALUATION COMPLETE ({duration:.1f}s)")
            self._logger.info("=" * 60)
        
        return self.metrics
