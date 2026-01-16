# Deployment

KladML provides robust tools for deploying your trained models to various environments, with a primary focus on Edge and Embedded devices via TorchScript optimization.

## Automatic Export

Starting from v0.3.0, every training run automatically generates a deployment-optimized artifact alongside the standard checkpoint.

When `train()` completes (or finds a best model), it creates:
- `checkpoints/best_model.pth`: Python-dependent PyTorch checkpoint (contains weights + optimizer state). Use for resuming training or fine-tuning.
- `checkpoints/best_model_jit.pt`: **Deployment Artifact**. A standalone TorchScript file containing the model architecture, weights, and embedded scaler statistics.

## Manual Export

You can export any existing checkpoint using the CLI:

```bash
kladml models export \
    --checkpoint ./data/projects/my_project/experiments/exp1/run_id/checkpoints/best_model.pth \
    --output ./deployment/model.pt \
    --config ./data/projects/my_project/experiments/exp1/run_id/config.yaml
```

## Using the Deployed Model (Edge/Production)

The exported `.pt` file is self-contained. You do **not** need the `kladml` library installed on the target device, only `torch`.

### Python (Edge/Server)

```python
import torch

# 1. Load Model
model = torch.jit.load("model.pt")
model.eval()

# 2. Extract embedded scalar stats (Zero external dependencies!)
mean = float(model.metadata()["scaler_mean"])
scale = float(model.metadata()["scaler_scale"])

# 3. Inference
# Input: [Batch, SeqLen, 1] (Raw values)
input_data = torch.randn(1, 60, 1) 

# Preprocess (Standard Scaling)
input_scaled = (input_data - mean) / scale

# Forward Pass
# Output: Mean, LogVar
pred_mean, pred_logvar = model(input_scaled)

print(f"Prediction: {pred_mean}")
```

### C++ (Embedded)

Since it is a TorchScript module, it can be loaded in C++ using `libtorch`:

```cpp
#include <torch/script.h>

int main() {
  torch::jit::script::Module module = torch::jit::load("model.pt");
  // ... inference logic ...
}
```

## Supported Architectures

- **Gluformer**: Fully supported. Wraps inputs to `[Batch, 60, 1]` and handles decoder inputs automatically.
