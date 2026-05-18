# Loss Card: TweedieLoss

## 1. Mathematical Intent
Statistical deviance loss based on the Tweedie distribution (1 < p < 2), which naturally handles zero-inflated continuous data with a point mass at zero and a Gamma-distributed tail.

## 2. Assumptions & Domain
- **Input Scale:** **STRICT REQUIREMENT.** Must receive count-scale or similar original magnitude data. Applying `MinMax` or aggressive `Log` before Tweedie may violate its mean-variance relationship assumptions.
- **Target Domain:** Non-negative continuous.
- **Positivity:** Strictly required for targets ($y \ge 0$). Predictions are passed through `softplus` to ensure $\mu > 0$.

## 3. Mandatory Genes (DNA)
- `p`: Power parameter (1.5 is standard for conflict).
- `non_zero_weight`: Base weight multiplier.
- `zero_threshold`: Threshold for internal weighting.
- `eps`: Stability constant.

## 4. Behavioral Profile (Audit Results)
- **Cowardice Signal:** **EXCELLENT (BRAVE).** Naturally resistant to mass collapse. Produced the highest mean prediction in the basin research suite.
- **Seed Sensitivity:** **LOW** (2.21 std). More stable attractor than threshold-based losses.
- **Gradient Flow:** Verified via `gradcheck`. Uses `softplus` link for numerical stability.

## 5. Known Failure Modes
- **Scale Violation:** If fed data that doesn't follow a Poisson-Gamma variance structure, optimization may fail to converge.
- **Numerical Sanity:** Raises `NumericalSanityError` on NaNs.

---
🖖 **"In this repository, a crash is a successful defense of scientific integrity."**
