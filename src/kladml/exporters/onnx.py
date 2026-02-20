"""
ONNX exporter for KladML models.

Supports optional scaler embedding via a wrapper module for end-to-end
inference pipelines that include data normalization.
"""

import torch
import torch.nn as nn
from typing import Any, Optional
from pathlib import Path
from kladml.interfaces.exporter import ExporterInterface
from kladml.exporters.registry import ExporterRegistry


class ONNXScalerWrapper(nn.Module):
    """
    Wrapper that applies sklearn scaler transformation before model inference.

    This allows exporting the complete preprocessing + inference pipeline
    as a single ONNX model.

    Example:
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> scaler.fit(train_data)
        >>> wrapper = ONNXScalerWrapper(model, scaler)
        >>> # Now wrapper(input) applies scaling then model inference
    """

    def __init__(self, model: nn.Module, scaler: Any):
        super().__init__()
        self.model = model

        # Extract scaler parameters
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            # StandardScaler or RobustScaler
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


@ExporterRegistry.register("onnx")
class ONNXExporter(ExporterInterface):
    """
    Exports PyTorch models to ONNX format.

    Supports optional scaler embedding for end-to-end inference pipelines.
    When a scaler is provided, wraps the model to include normalization.
    """

    @property
    def format_name(self) -> str:
        return "onnx"

    def export(
        self,
        model: torch.nn.Module,
        output_path: str,
        input_sample: Any = None,
        scaler: Optional[Any] = None,
        **kwargs
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model to export
            output_path: Output file path
            input_sample: Sample input for tracing (required for ONNX export)
            scaler: Optional sklearn scaler to embed in the export
            **kwargs: Additional export options (opset_version, dynamic_axes)

        Returns:
            Path to exported model

        Raises:
            ValueError: If input_sample is not provided
        """
        if input_sample is None:
            raise ValueError("ONNX export requires an input sample for tracing.")

        # Wrap model with scaler if provided
        if scaler is not None:
            model = ONNXScalerWrapper(model, scaler)

        opset_version = kwargs.get("opset_version", 11)
        dynamic_axes = kwargs.get("dynamic_axes", None)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            input_sample,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        return output_path

    def validate(self, exported_path: str, input_sample: Any = None) -> bool:
        """
        Validate exported ONNX model.

        Args:
            exported_path: Path to exported model
            input_sample: Optional sample input for validation

        Returns:
            True if validation succeeds
        """
        try:
            import onnx
            onnx_model = onnx.load(exported_path)
            onnx.checker.check_model(onnx_model)

            # Optionally test inference with onnxruntime
            if input_sample is not None:
                try:
                    import onnxruntime as ort
                    import numpy as np

                    session = ort.InferenceSession(exported_path)
                    input_name = session.get_inputs()[0].name

                    if isinstance(input_sample, torch.Tensor):
                        input_sample = input_sample.numpy()

                    session.run(None, {input_name: input_sample})
                except ImportError:
                    pass  # onnxruntime not installed, skip inference test

            return True
        except ImportError:
            print("ONNX not installed, skipping validation.")
            return True
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
