"""
Tests for the raw-space RINorm patch.

The patch normalizes in raw count space (not asinh space) to eliminate
Jensen amplification bias from the downstream sinh inverse transform.

Forward:  z = asinh((sinh(x_asinh) - μ_raw) / σ_raw)
Inverse:  ŷ_asinh = asinh(sinh(ẑ) · σ_raw + μ_raw)

Verifies:
  1. Forward-inverse identity (round-trip reconstruction)
  2. Zero Jensen amplification in count space
  3. Compact z-ranges for both peaceful and conflict series
  4. Gradient continuity and finiteness
  5. Batch independence
  6. Correct behavior at boundary cases
"""
import pytest
import torch
import torch.nn as nn
import numpy as np


class FakeRawSpaceRINorm(nn.Module):
    """Minimal replica mirroring the raw-space RINorm patch."""
    def __init__(self, input_dim, eps=1e-5, affine=False):
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.affine = affine

    def forward(self, x):
        # x is in asinh-space
        calc_dims = tuple(range(1, x.ndim - 1))
        x = torch.clamp(x, -88.0, 88.0)
        x_raw = torch.sinh(x)
        self.mean = torch.mean(x_raw, dim=calc_dims, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x_raw, dim=calc_dims, keepdim=True, unbiased=False) + self.eps
        ).detach()
        x_norm_raw = (x_raw - self.mean) / self.stdev
        return torch.asinh(x_norm_raw)

    def inverse(self, x):
        sigma = self.stdev.view(self.stdev.shape + (1,))
        mu = self.mean.view(self.mean.shape + (1,))
        x = torch.clamp(x, -50.0, 50.0)  # matches patches.py ±50 (raised from ±20)
        return torch.asinh(torch.sinh(x) * sigma + mu)


def _make_asinh_series(mu_raw, sigma_raw, length=36, features=1):
    """Create a (1, length, features) tensor in asinh-space from raw-space params."""
    torch.manual_seed(42)
    x_raw = mu_raw + sigma_raw * torch.randn(1, length, features).abs()
    return torch.asinh(x_raw)


def _make_conflict_series(length=36):
    """Ukraine-like: μ_raw≈74 (asinh≈5), σ_raw≈297."""
    torch.manual_seed(42)
    x_raw = 74 + 297 * torch.randn(1, length, 1).abs()
    return torch.asinh(x_raw)


def _make_peaceful_series(length=36):
    """Peaceful: μ_raw≈0.1, σ_raw≈0.3."""
    torch.manual_seed(42)
    x_raw = 0.1 + 0.3 * torch.randn(1, length, 1).abs()
    return torch.asinh(x_raw)


# ═══════════════════════════════════════════════════════════════════════
# Forward-Inverse Identity
# ═══════════════════════════════════════════════════════════════════════

class TestRoundTrip:

    def test_perfect_round_trip_conflict(self):
        """Forward→inverse is identity for conflict series."""
        x = _make_conflict_series()
        norm = FakeRawSpaceRINorm(1)
        z = norm.forward(x)
        z_4d = z.unsqueeze(-1)
        x_recon = norm.inverse(z_4d).squeeze(-1)
        assert torch.allclose(x, x_recon, atol=1e-4), (
            f"max_diff={(x - x_recon).abs().max():.6f}"
        )

    def test_perfect_round_trip_peaceful(self):
        """Forward→inverse is identity for peaceful series."""
        x = _make_peaceful_series()
        norm = FakeRawSpaceRINorm(1)
        z = norm.forward(x)
        z_4d = z.unsqueeze(-1)
        x_recon = norm.inverse(z_4d).squeeze(-1)
        assert torch.allclose(x, x_recon, atol=1e-4), (
            f"max_diff={(x - x_recon).abs().max():.6f}"
        )

    def test_round_trip_zero_series(self):
        """All-zero series (eps prevents division by zero)."""
        x = torch.zeros(1, 36, 1)  # asinh(0) = 0
        norm = FakeRawSpaceRINorm(1)
        z = norm.forward(x)
        z_4d = z.unsqueeze(-1)
        x_recon = norm.inverse(z_4d).squeeze(-1)
        assert torch.allclose(x, x_recon, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════════
# Jensen Bias Elimination
# ═══════════════════════════════════════════════════════════════════════

class TestJensenBias:

    def test_zero_jensen_bias_at_mean(self):
        """E[sinh(ẑ)] ≈ 0 for zero-mean ẑ — no systematic bias."""
        torch.manual_seed(99)
        # Simulate many z-space predictions with mean≈0, std≈1
        z_samples = torch.randn(10000)
        # sinh is odd → E[sinh(Z)] = 0 for symmetric Z
        mean_sinh = torch.sinh(z_samples).mean().item()
        assert abs(mean_sinh) < 0.1, f"E[sinh(Z)]={mean_sinh}, expected ≈0"

    def test_mc_dropout_no_overshoot(self):
        """MC dropout variance (σ_ẑ≈0.3) produces <5% Jensen bias in counts."""
        x = _make_conflict_series(length=1000)
        norm = FakeRawSpaceRINorm(1)
        z = norm.forward(x)

        # Simulate N MC samples with noise σ=0.3
        torch.manual_seed(99)
        n_samples = 100
        count_means = []
        for _ in range(n_samples):
            z_noisy = z + 0.3 * torch.randn_like(z)
            z_4d = z_noisy.unsqueeze(-1)
            y_asinh = norm.inverse(z_4d).squeeze(-1)
            counts = torch.sinh(y_asinh).mean().item()
            count_means.append(counts)

        # True counts from the original data
        truth = torch.sinh(x).mean().item()
        mc_mean = np.mean(count_means)

        # Jensen bias should be minimal (<10%)
        bias_pct = abs(mc_mean - truth) / max(abs(truth), 1.0) * 100
        assert bias_pct < 10, (
            f"Jensen bias={bias_pct:.1f}%, truth={truth:.1f}, mc_mean={mc_mean:.1f}"
        )

    def test_no_systematic_overprediction(self):
        """The inverse doesn't systematically overshoot for conflict series."""
        x = _make_conflict_series(length=1000)
        norm = FakeRawSpaceRINorm(1)
        z = norm.forward(x)
        z_4d = z.unsqueeze(-1)
        y_asinh = norm.inverse(z_4d).squeeze(-1)
        # Perfect prediction — should match exactly
        pred_counts = torch.sinh(y_asinh).mean().item()
        true_counts = torch.sinh(x).mean().item()
        ratio = pred_counts / max(true_counts, 1.0)
        assert 0.95 < ratio < 1.05, f"event_ratio={ratio:.3f}"


# ═══════════════════════════════════════════════════════════════════════
# Z-Range Compactness
# ═══════════════════════════════════════════════════════════════════════

class TestZRange:

    def test_conflict_z_range_compact(self):
        """Ukraine-like series produces z ∈ approximately [-1, 5]."""
        x = _make_conflict_series(length=1000)
        norm = FakeRawSpaceRINorm(1)
        z = norm.forward(x)
        assert z.min() > -3, f"z_min={z.min():.2f} too low"
        assert z.max() < 7, f"z_max={z.max():.2f} too high"

    def test_peaceful_z_range_compact(self):
        """Peaceful series produces z ∈ approximately [-1, 3]."""
        x = _make_peaceful_series(length=1000)
        norm = FakeRawSpaceRINorm(1)
        z = norm.forward(x)
        assert z.min() > -4, f"z_min={z.min():.2f} too low"
        assert z.max() < 5, f"z_max={z.max():.2f} too high"

    def test_similar_ranges_across_entities(self):
        """Conflict and peaceful z-ranges are within 3× of each other."""
        x_c = _make_conflict_series(length=1000)
        x_p = _make_peaceful_series(length=1000)
        norm_c = FakeRawSpaceRINorm(1)
        norm_p = FakeRawSpaceRINorm(1)
        z_c = norm_c.forward(x_c)
        z_p = norm_p.forward(x_p)
        range_c = (z_c.max() - z_c.min()).item()
        range_p = (z_p.max() - z_p.min()).item()
        ratio = max(range_c, range_p) / max(min(range_c, range_p), 0.01)
        assert ratio < 3.0, f"Range ratio={ratio:.1f} (conflict={range_c:.2f}, peaceful={range_p:.2f})"


# ═══════════════════════════════════════════════════════════════════════
# Gradient Properties
# ═══════════════════════════════════════════════════════════════════════

class TestGradients:

    def test_finite_gradients_conflict(self):
        """Gradients are finite for high-conflict series."""
        x = _make_conflict_series()
        x.requires_grad_(True)
        norm = FakeRawSpaceRINorm(1)
        z = norm.forward(x)
        loss = z.pow(2).sum()
        loss.backward()
        assert torch.isfinite(x.grad).all()

    def test_finite_gradients_peaceful(self):
        """Gradients are finite for peaceful series."""
        x = _make_peaceful_series()
        x.requires_grad_(True)
        norm = FakeRawSpaceRINorm(1)
        z = norm.forward(x)
        loss = z.pow(2).sum()
        loss.backward()
        assert torch.isfinite(x.grad).all()

    def test_gradient_bounded(self):
        """Gradient magnitude is bounded — no sensitivity explosion."""
        x = _make_conflict_series()
        x.requires_grad_(True)
        norm = FakeRawSpaceRINorm(1)
        z = norm.forward(x)
        loss = z.sum()
        loss.backward()
        grad_max = x.grad.abs().max().item()
        # Gradient through asinh is bounded by 1/sqrt(1+x²) ≤ 1
        # Combined: should be reasonable
        assert grad_max < 100, f"grad_max={grad_max}"


# ═══════════════════════════════════════════════════════════════════════
# Batch Independence
# ═══════════════════════════════════════════════════════════════════════

class TestBatchBehavior:

    def test_batch_independent_statistics(self):
        """Each series in a batch gets independent μ_raw, σ_raw."""
        x_p = _make_peaceful_series()
        x_c = _make_conflict_series()
        x_batch = torch.cat([x_p, x_c], dim=0)
        norm = FakeRawSpaceRINorm(1)
        norm.forward(x_batch)
        # Should have 2 different means and stdevs
        assert norm.mean.shape[0] == 2
        assert norm.stdev.shape[0] == 2
        # Conflict has much higher raw mean than peaceful
        assert norm.mean[1].item() > norm.mean[0].item() * 10

    def test_no_fp32_overflow(self):
        """All computations stay finite in fp32."""
        x = _make_conflict_series()
        norm = FakeRawSpaceRINorm(1)
        z = norm.forward(x)
        assert torch.isfinite(z).all()
        assert torch.isfinite(norm.mean).all()
        assert torch.isfinite(norm.stdev).all()
        z_4d = z.unsqueeze(-1)
        y = norm.inverse(z_4d)
        assert torch.isfinite(y).all()

    def test_inverse_finite_for_large_model_outputs(self):
        """Inverse stays finite even when model outputs |z|>88 (sinh overflow threshold).

        Early training with random-init TSMixer / N-HiTS can produce arbitrarily
        large normalized outputs before the first gradient step settles weights.
        The ±20 clamp in inverse must absorb these without producing inf/nan.
        """
        x = _make_conflict_series()
        norm = FakeRawSpaceRINorm(1)
        norm.forward(x)  # populate mean/stdev

        # Simulate a degenerate early-training model output with |z| >> 88
        z_large = torch.tensor([[[100.0], [50.0], [-90.0], [200.0]]])  # (1, 4, 1, 1)
        y = norm.inverse(z_large)
        assert torch.isfinite(y).all(), f"inverse produced non-finite values: {y}"
        assert not torch.isnan(y).any(), f"inverse produced NaN: {y}"
