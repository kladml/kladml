"""
TorchScript exporter for KladML models.

Supports optional scaler embedding via a wrapper module for end-to-end
inference pipelines that include data normalization.
"""

import torch
import torch.nn as nn
from typing import Any, Optional
from pathlib import Path
from kladml.interfaces.exporter import ExporterInterface
from kladml.exporters.registry import ExporterRegistry


class ScalerWrapper(nn.Module):
    """
    Wrapper that applies sklearn scaler transformation before model inference.

    This allows exporting the complete preprocessing + inference pipeline
    as a single TorchScript module.

    Example:
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> scaler.fit(train_data)
        >>> wrapper = ScalerWrapper(model, scaler)
        >>> # Now wrapper(input) applies scaling then model inference
    """

    def __init__(self, model: nn.Module, scaler: Any):
        super().__init__()
        self.model = model

        # Extract scaler parameters
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            self.register_buffer('mean', torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer('scale', torch.tensor(scaler.scale_, dtype=torch.float32))
        elif hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):
            # MinMaxScaler
            self.register_buffer('mean', torch.tensor(scaler.min_, dtype=torch.float32))
            self.register_buffer('scale', torch.tensor(scaler.scale_, dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported scaler type: {type(scaler)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaling then model inference."""
        # Normalize: (x - mean) / scale
        x_normalized = (x - self.mean) / self.scale
        return self.model(x_normalized)


@ExporterRegistry.register("torchscript")
class TorchScriptExporter(ExporterInterface):
    """
    Exports PyTorch models to TorchScript (JIT).

    Supports optional scaler embedding for end-to-end inference pipelines.
    When a scaler is provided, wraps the model to include normalization.
    """

    @property
    def format_name(self) -> str:
        return "torchscript"

    def export(
        self,
        model: torch.nn.Module,
        output_path: str,
        input_sample: Any = None,
        scaler: Optional[Any] = None,
        **kwargs
    ) -> str:
        """
        Export model to TorchScript format.

        Args:
            model: PyTorch model to export
            output_path: Output file path
            input_sample: Sample input for tracing (required for most models)
            scaler: Optional sklearn scaler to embed in the export
            **kwargs: Additional export options

        Returns:
            Path to exported model
        """
        # Wrap model with scaler if provided
        if scaler is not None:
            model = ScalerWrapper(model, scaler)

        if input_sample is not None:
            # Trace mode (preferred for models with consistent control flow)
            scripted_model = torch.jit.trace(model, input_sample)
        else:
            # Script mode (for models with dynamic control flow)
            scripted_model = torch.jit.script(model)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        scripted_model.save(output_path)
        return output_path

    def validate(self, exported_path: str, input_sample: Any = None) -> bool:
        """
        Validate exported TorchScript model.

        Args:
            exported_path: Path to exported model
            input_sample: Optional sample input for validation

        Returns:
            True if validation succeeds
        """
        try:
            loaded = torch.jit.load(exported_path)
            loaded.eval()
            if input_sample is not None:
                with torch.no_grad():
                    loaded(input_sample)
            return True
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
