import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_batch(
    n_series: int,
    horizon: int,
    n_channels: int,
    event_prob: float,
    base_noise: float,
    spike_scale: float,
    spike_sigma: float,
    seed: int,
) -> np.ndarray:
    """Generate synthetic multichannel batch in transformed space.

    Output shape: [B, T, C]
    """
    rng = np.random.default_rng(seed)
    y = rng.normal(loc=0.0, scale=base_noise, size=(n_series, horizon, n_channels)).astype(np.float32)

    event_mask = rng.random((n_series, horizon, n_channels)) < event_prob
    spikes = rng.lognormal(mean=np.log(spike_scale), sigma=spike_sigma, size=(n_series, horizon, n_channels)).astype(np.float32)

    # Randomize spike sign slightly to mimic transformed residual dynamics.
    signs = rng.choice(np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float32), size=(n_series, horizon, n_channels))
    y += event_mask.astype(np.float32) * spikes * signs

    # Keep values within realistic transformed-space bounds.
    y = np.clip(y, -8.0, 12.0)
    return y


def make_baseline_prediction(y_true: np.ndarray, pred_noise: float, seed: int) -> np.ndarray:
    """A templated baseline prediction leaning toward near-zero median behavior."""
    rng = np.random.default_rng(seed)
    per_series_mean = y_true.mean(axis=1, keepdims=True)

    # Under-reactive predictor: mostly low-variance template around damped mean.
    y_pred = 0.35 * per_series_mean + rng.normal(0.0, pred_noise, size=y_true.shape).astype(np.float32)
    y_pred = np.clip(y_pred, -6.0, 8.0)
    return y_pred


def summarize(name: str, y_true: np.ndarray, threshold: float) -> dict:
    abs_y = np.abs(y_true)
    return {
        "name": name,
        "shape": list(y_true.shape),
        "mean": float(y_true.mean()),
        "std": float(y_true.std()),
        "median": float(np.median(y_true)),
        "p95_abs": float(np.percentile(abs_y, 95)),
        "p99_abs": float(np.percentile(abs_y, 99)),
        "event_frac_gt_tau": float((abs_y > threshold).mean()),
        "zeroish_frac_abs_lt_0p05": float((abs_y < 0.05).mean()),
    }


def to_long_df(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str) -> pd.DataFrame:
    b, t, c = y_true.shape
    idx = np.indices((b, t, c))
    df = pd.DataFrame(
        {
            "series_id": idx[0].ravel().astype(np.int32),
            "step": idx[1].ravel().astype(np.int16),
            "channel_id": idx[2].ravel().astype(np.int8),
            "y_true": y_true.ravel().astype(np.float32),
            "y_pred_template": y_pred.ravel().astype(np.float32),
            "dataset": dataset_name,
        }
    )
    return df


def main() -> None:
    out_dir = Path("/Users/dylanpinheiro/Desktop/views-platform/views-r2darts2/data/synthetic_loss_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    tau = 0.88
    n_channels = 3
    horizon = 36

    # PGM-like: much sparser with heavier tails.
    pgm_true = make_batch(
        n_series=512,
        horizon=horizon,
        n_channels=n_channels,
        event_prob=0.035,
        base_noise=0.03,
        spike_scale=1.8,
        spike_sigma=1.0,
        seed=20260717,
    )
    pgm_pred = make_baseline_prediction(pgm_true, pred_noise=0.06, seed=20260718)

    # CM-like: denser with milder tails.
    cm_true = make_batch(
        n_series=512,
        horizon=horizon,
        n_channels=n_channels,
        event_prob=0.11,
        base_noise=0.05,
        spike_scale=1.2,
        spike_sigma=0.7,
        seed=20260719,
    )
    cm_pred = make_baseline_prediction(cm_true, pred_noise=0.08, seed=20260720)

    np.savez_compressed(out_dir / "pgm_batch.npz", y_true=pgm_true, y_pred_template=pgm_pred)
    np.savez_compressed(out_dir / "cm_batch.npz", y_true=cm_true, y_pred_template=cm_pred)

    df = pd.concat(
        [
            to_long_df(pgm_true, pgm_pred, "pgm"),
            to_long_df(cm_true, cm_pred, "cm"),
        ],
        ignore_index=True,
    )
    df.to_parquet(out_dir / "synthetic_pgm_cm_long.parquet", index=False)

    summary = {
        "tau": tau,
        "pgm": summarize("pgm", pgm_true, tau),
        "cm": summarize("cm", cm_true, tau),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote synthetic data to", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
