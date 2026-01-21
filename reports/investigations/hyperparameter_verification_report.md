# Hyperparameter Verification Report

This document tracks the systematic verification of training hyperparameters based on the memo "Verification and Validation of Trainer, Optimizer, and Control Hyperparameters".

---

## 1. Optimizer & Weight Decay

### Parameter Under Review
- `optimizer_cls: Adam`
- `weight_decay: 0.0003`

### Verification Method
- An ablation test was conducted using a simple `NLinearModel` on synthetic `log1p`-transformed data with `ShrinkageLoss` and a learning rate of `1e-3`.
- Two runs were performed: one with `weight_decay = 0.0003` and one with `weight_decay = 0.0`.

### Findings
- **`weight_decay=0.0003`**: Val Loss: `0.1668`, Mag Ratio: `4.6179`
- **`weight_decay=0.0`**: Val Loss: `0.1801`, Mag Ratio: `3.6580`

### Conclusion
- The use of `weight_decay=0.0003` is deemed reasonable for now as it doesn't seem to be the root cause of the original underprediction problem. The discrepancy in magnitude bias between this test and the real-world scenario suggests the cause may lie with the `N-BEATS` model's complexity or its interaction with the real data distribution.

---

## 2. Gradient Clipping

### Parameter Under Review
- `gradient_clip_val: 0.64`

### Verification Method
- A custom `PyTorch Lightning` callback (`ClippingMonitor`) was created to track the frequency of gradient clipping during a training run.
- A single run was performed with the best-known configuration (`ShrinkageLoss`, `log1p`, `lr=0.001`).

### Findings
- Clipping occurred in **1 out of 2150 steps (0.05%)**.

### Conclusion
- The use of `gradient_clip_val: 0.64` is **not warranted** for this configuration. The training is stable enough that the natural gradient norm almost never exceeds this threshold. The parameter is currently "dead" and adds unnecessary complexity. It could be removed.

---

## 3. LR Scheduler & Early Stopping

### Parameters Under Review
- `lr_scheduler_cls: ReduceLROnPlateau` (patience: 7, factor: 0.46)
- `early_stopping_patience: 20`
- `early_stopping_min_delta: 0.001`

### Verification Method
- A custom `PyTorch Lightning` callback (`LRSchedulerMonitor`) was created to log `val_loss` and `lr` at every epoch during a long training run (100 epochs).

### Findings
- The `ReduceLROnPlateau` scheduler correctly reduced the learning rate twice (at epochs 12 and 20) after `val_loss` failed to improve for 7 epochs.
- The `EarlyStopping` callback correctly terminated the run at epoch 23, after the `val_loss` had not seen a significant improvement for 20 epochs.

### Conclusion
- The scheduler and early stopping mechanisms are **correctly implemented, active, and well-balanced**. The scheduler has enough time to attempt to break out of plateaus before training is halted. The configuration is **warranted and prudent**.

---
