# Loss Card: WeightedPenaltyHuberLoss

## 1. Mathematical Intent
A custom variant of the Huber loss designed specifically for zero-inflated conflict data. It applies multiplicative penalties to asymmetric error types (False Positives and False Negatives).

## 2. Assumptions & Domain
- **Input Scale:** Works best on `AsinhTransform` or `LogTransform` scales.
- **Target Domain:** Non-negative continuous or count data.
- **Positivity:** Not strictly required but mathematically assumes targets represent magnitude.

## 3. Mandatory Genes (DNA)
- `zero_threshold`: Decision boundary for "conflict vs no-conflict".
- `delta`: Quadratic-to-linear transition point.
- `non_zero_weight`: Base multiplier for actual conflict months.
- `false_positive_weight`: Multiplier for over-prediction.
- `false_negative_weight`: Multiplier for under-prediction (The "Brave" knob).

## 4. Behavioral Profile (Audit Results)
- **Cowardice Signal:** Low to Medium. Encourages bravery via `false_negative_weight`, but sensitive to the `zero_threshold`.
- **Seed Sensitivity:** **HIGH** (3.79 std). Hard thresholds create a discontinuous gradient landscape that can lead to basin bifurcation.
- **Gradient Flow:** Verified via `gradcheck`. Valid gradients exist across the quadratic and linear regions.

## 5. Known Failure Modes
- **Threshold Jitter:** If the model oscillates around the `zero_threshold`, gradients can become unstable.
- **Numerical Sanity:** Raises `NumericalSanityError` on NaNs (Mandatory Fortress Gate).

---
🖖 **"In this repository, a crash is a successful defense of scientific integrity."**
