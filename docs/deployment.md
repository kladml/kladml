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

---

## 🔮 Understanding Output: Aleatoric Uncertainty

The model outputs two tensors:
1.  **`pred_mean`**: The predicted glucose value.
2.  **`pred_logvar`**: The predicted **Log-Variance** of the error.

This represents **Aleatoric Uncertainty** (data noise). The model is saying: *"I predict X, but I'm Y% unsure because the input data is noisy/ambiguous."*

### Converting Log-Variance to Confidence Intervals

To get the Standard Deviation ($\sigma$) and 95% Confidence Interval ($CI$):

$$ \sigma = \sqrt{\exp(\text{logvar})} $$
$$ CI_{95\%} = [\text{mean} - 1.96\sigma, \text{mean} + 1.96\sigma] $$

> [!TIP]
> Always assume Gaussian distribution for the error.

---

## 💻 Visualization in React (Frontend)

To visualize the prediction with its uncertainty "cloud", you can use libraries like **Recharts**.

### 1. Data Processing (JavaScript)

```javascript
// Function to process model output
const processPredictions = (means, logvars) => {
  return means.map((mean, i) => {
    const logvar = logvars[i];
    const sigma = Math.sqrt(Math.exp(logvar));
    
    return {
      timestamp: Date.now() + i * 5 * 60 * 1000, // +5 mins per step
      glucose: mean,
      range: [mean - 1.96 * sigma, mean + 1.96 * sigma] // [Low, High]
    };
  });
};
```

### 2. React Component (Recharts)

Displaying the confidence interval as an `Area` behind the `Line`.

```jsx
import { ComposedChart, Line, Area, XAxis, YAxis, Tooltip } from 'recharts';

const GlucoseChart = ({ data }) => (
  <ComposedChart width={600} height={300} data={data}>
    <XAxis dataKey="timestamp" tickFormatter={(t) => new Date(t).toLocaleTimeString()} />
    <YAxis domain={['auto', 'auto']} />
    <Tooltip />
    
    {/* Confidence Interval (Gray Cloud) */}
    <Area 
      type="monotone" 
      dataKey="range" 
      fill="#8884d8" 
      stroke="#8884d8" 
      opacity={0.3} 
    />
    
    {/* Predicted Value (Solid Line) */}
    <Line 
      type="monotone" 
      dataKey="glucose" 
      stroke="#8884d8" 
      strokeWidth={3} 
      dot={false} 
    />
  </ComposedChart>
);
```
