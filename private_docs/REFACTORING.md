# KladML Codebase Modernization Plan

This document outlines the steps to modernize the KladML codebase to 2026 standards.

## 1. Modern Type Hinting (Python 3.10+) 🧹
**Goal:** Remove `typing` imports in favor of built-in generics and union operators.

- [ ] Replace `List[...]`, `Dict[...]`, `Tuple[...]` with `list[...]`, `dict[...]`, `tuple[...]`.
- [ ] Replace `Optional[T]` and `Union[A, B]` with `T | None` and `A | B`.
- [ ] Remove `from typing import List, Dict, Optional, Union`.
- [ ] Tooling: Use `ruff` or `pyupgrade --py310-plus` to automate this.

## 2. Typed Configuration (Pydantic v2) 🛡️
**Goal:** Replace raw `Dict[str, Any]` configs with validated Pydantic models. (Completed v0.7.0) ✅

- [ ] Create `src/kladml/config/schema.py` defining base configs:
    ```python
    class TrainingConfig(BaseModel):
        epochs: int = Field(default=10, ge=1)
        batch_size: int = 32
        learning_rate: float = 1e-3
    ```
- [ ] Update `UniversalTrainer` to accept `config: TrainingConfig`.
- [ ] Update `BaseModel.train()` signature to use typed config.
- [ ] Benefits: Auto-validation, IDE autocomplete, JSON schema generation for UI.

## 3. Modern Training Loop (Accelerate / Fabric) ⚡
**Goal:** Remove manual device placement boiler-plate (`.to(device)`, `if cuda...`) in `UniversalTrainer`.

- [ ] Introduce **HuggingFace Accelerate** or **Lightning Fabric**.
- [ ] Refactor `trainer.py`:
    ```python
    # Before
    loss.backward()
    optimizer.step()
    
    # After (Accelerate)
    accelerator.backward(loss)
    optimizer.step()
    ```
- [ ] Enables: Mixed Precision (FP16/BF16), Multi-GPU (DDP), DeepSpeed integration directly.

## 4. Pathlib Standardization 📂
**Goal:** Eliminate string-based path manipulation.

- [ ] Audit code for `os.path.join`, `os.path.exists`.
- [ ] Enforce `pathlib.Path` in all function signatures.
- [ ] Ensure `resolve_dataset_path` always returns a `Path` object.

## 5. Structured Logging (Loguru / Structlog) 📝
**Goal:** Replace standard `logging` with structured, context-aware logging.

- [ ] Replace `logging` with `loguru` (simpler) or `structlog` (more enterprise).
- [ ] JSON output option for production/CloudWatch.
- [ ] Rich-formatted console output for local development.
- [ ] Context binding: `logger = logger.bind(run_id=run_id)` -> automatic run_id in all logs.

---

## Execution Strategy

1. **Step 1 (Low Risk):** Type Hinting Refactor (Automated).
2. **Step 2 (Medium Risk):** Pathlib & Logging.
3. **Step 3 (High Impact):** Typed Configuration (Requires API changes).
4. **Step 4 (High Risk):** Training Loop Rewrite (Core logic change).
