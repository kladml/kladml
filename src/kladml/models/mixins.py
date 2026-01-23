
import os
import logging
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)

class TorchExportMixin:
    """
    Mixin class to add TorchScript export capabilities to any model.
    Requires the class to have 'model' (nn.Module) and 'device' attributes.
    """

    def export_torchscript(self, output_path: str, input_shape: Optional[tuple] = None) -> None:
        """
        Export the model to TorchScript (traced) for no-code deployment.
        
        Args:
            output_path: Where to save the exported model (.pt or .onnx)
            input_shape: Tuple of (seq_len, num_features). If None, tries to auto-detect.
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model not initialized. Ensure 'self.model' is set.")
        
        # Determine device (fallback to cpu if not present)
        device = getattr(self, "device", "cpu")
            
        self.model.eval()
        
        # 1. Determine Input Shape
        if input_shape is None:
            # Try to infer from attributes common in KladML models
            seq_len = getattr(self, "seq_len", getattr(self.config, "seq_len", None))
            num_features = getattr(self, "num_features", getattr(self.config, "num_features", None))
            
            # Gluformer legacy attribute support
            if num_features is None:
                 num_features = getattr(self, "enc_in", None)

            if seq_len and num_features:
                # Add default batch size of 1
                try:
                    dummy_input = torch.randn(1, seq_len, num_features).to(device)
                except Exception as e:
                     # Check if num_features/seq_len are actually ints
                    logger.warning(f"Auto-shape failed ({e}). Trying to cast dimensions...")
                    dummy_input = torch.randn(1, int(seq_len), int(num_features)).to(device)
            else:
                raise ValueError(
                    "Could not determine input shape automatically. "
                    "Please provide input_shape=(seq_len, num_features)."
                )
        else:
            dummy_input = torch.randn(1, *input_shape).to(device)
            
        # 2. Trace
        try:
            logger.info(f"Tracing model with input shape {dummy_input.shape}...")
            # Use strict=False to be permissive with dict outputs or unused layers
            traced_model = torch.jit.trace(self.model, dummy_input)
            
            # 3. Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            traced_model.save(output_path)
            logger.info(f"✅ Model successfully exported to TorchScript: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export TorchScript: {e}")
            raise

    def export_onnx(self, output_path: str, input_shape: Optional[tuple] = None, validate: bool = True) -> None:
        """
        Export the model to ONNX format for universal deployment.
        
        Args:
            output_path: Where to save the exported model (.onnx)
            input_shape: Tuple of (seq_len, num_features). If None, tries to auto-detect.
            validate: If True, validates the exported model after saving.
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model not initialized. Ensure 'self.model' is set.")
        
        device = getattr(self, "device", "cpu")
        self.model.eval()
        
        # 1. Determine Input Shape
        if input_shape is None:
            seq_len = getattr(self, "seq_len", getattr(self.config, "seq_len", None))
            num_features = getattr(self, "num_features", getattr(self.config, "num_features", None))
            if num_features is None:
                num_features = getattr(self, "enc_in", None)
            
            if seq_len and num_features:
                dummy_input = torch.randn(1, int(seq_len), int(num_features)).to(device)
            else:
                raise ValueError("Could not determine input shape. Provide input_shape=(seq_len, num_features).")
        else:
            dummy_input = torch.randn(1, *input_shape).to(device)
        
        # 2. Export to ONNX
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        logger.info(f"Exporting to ONNX with input shape {dummy_input.shape}...")
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,  # 17+ for LayerNormalization support
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        logger.info(f"✅ ONNX export complete: {output_path}")
        
        # 3. Validate
        if validate:
            self._validate_onnx_export(output_path, dummy_input)
    
    def _validate_onnx_export(self, onnx_path: str, test_input: torch.Tensor) -> None:
        """Validate ONNX export by comparing outputs."""
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np
            
            # Check model structure
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Compare outputs
            with torch.no_grad():
                pytorch_out = self.model(test_input).cpu().numpy()
            
            ort_session = ort.InferenceSession(onnx_path)
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
            ort_out = ort_session.run(None, ort_inputs)[0]
            
            if np.allclose(pytorch_out, ort_out, atol=1e-5):
                logger.info("✅ ONNX validation passed: outputs match PyTorch.")
            else:
                max_diff = np.max(np.abs(pytorch_out - ort_out))
                logger.warning(f"⚠️ ONNX validation: outputs differ (max diff: {max_diff:.6f})")
                
        except ImportError:
            logger.warning("onnx/onnxruntime not installed. Skipping validation.")
        except Exception as e:
            logger.warning(f"ONNX validation failed: {e}")
    
    def export_model(self, path: str, format: str = "onnx", **kwargs) -> None:
        """
        Export model to specified format.
        
        Args:
            path: Output file path.
            format: "onnx" or "torchscript".
        """
        if format == "onnx":
            self.export_onnx(path, **kwargs)
        elif format == "torchscript":
            self.export_torchscript(path, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}. Use 'onnx' or 'torchscript'.")
