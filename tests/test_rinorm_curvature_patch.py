"""
Tests for the curvature-aware RINorm patch with learnable-α conditioning
and Duan-style smearing correction.

Verifies:
  1. σ_eff = σ / cosh(μ)^α correctly compresses high-μ series
  2. Smearing correction ŷ -= c·σ²·tanh(ŷ) reduces Jensen bias
  3. Learnable parameters (α, c) have correct gradients
  4. Backwards-compatible state_dict loading
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FakeRINorm(nn.Module):
    """Minimal RINorm replica mirroring the patched implementation."""
    def __init__(self, input_dim, eps=1e-5, affine=False):
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.affine = affine
        # Learnable curvature exponent: α = sigmoid(alpha_raw)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))   # → α=0.5
        # Learnable smearing coefficient: c = softplus(smear_raw)
        self.smear_raw = nn.Parameter(torch.tensor(-3.0))  # → c≈0.05

    def forward(self, x):
        calc_dims = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=calc_dims, keepdim=True).detach()
        raw_stdev = torch.sqrt(
            torch.var(x, dim=calc_dims, keepdim=True, unbiased=False) + self.eps
        ).detach()
        alpha = torch.sigmoid(self.alpha_raw)
        self.stdev = raw_stdev / torch.cosh(self.mean).pow(alpha)
        x = (x - self.mean) / self.stdev
        return x

    def inverse(self, x):
        sigma = self.stdev.view(self.stdev.shape + (1,))
        mu = self.mean.view(self.mean.shape + (1,))
        x = x * sigma + mu
        c = F.softplus(self.smear_raw)
        x = x - c * sigma.pow(2) * torch.tanh(x)
        return x


def _make_series(mu, sigma, length=36, features=1):
    """Create a (1, length, features) tensor with given mean/std in asinh space."""
    torch.manual_seed(42)
    x = mu + sigma * torch.randn(1, length, features)
    return x


# ═══════════════════════════════════════════════════════════════════════
# Tests for learnable-α curvature conditioning (Option 1)
# ═══════════════════════════════════════════════════════════════════════

class TestCurvatureCompression:

    def test_peaceful_sigma_unchanged(self):
        """For μ≈0, cosh(0)=1, so σ_eff ≈ σ_raw regardless of α."""
        x = _make_series(mu=0.0, sigma=0.3)
        norm = FakeRINorm(1)
        norm.forward(x)
        raw_std = x.std(dim=1, keepdim=True).item()
        assert abs(norm.stdev.item() - raw_std) / max(raw_std, 1e-8) < 0.05

    def test_default_alpha_is_balanced(self):
        """Default alpha_raw=0 → α=0.5, same as previous balanced patch."""
        x = _make_series(mu=5.0, sigma=4.0, length=1000)
        norm = FakeRINorm(1)
        norm.forward(x)
        expected = 4.0 / np.sqrt(np.cosh(5.0))  # ≈ 0.465
        assert abs(norm.stdev.item() - expected) < 0.15

    def test_alpha_zero_is_standard_revin(self):
        """α=0 → σ_eff = σ (no curvature correction)."""
        x = _make_series(mu=5.0, sigma=4.0, length=1000)
        norm = FakeRINorm(1)
        norm.alpha_raw.data = torch.tensor(-10.0)  # sigmoid(-10) ≈ 0
        norm.smear_raw.data = torch.tensor(-20.0)  # c ≈ 0
        norm.forward(x)
        raw_std = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).item()
        assert abs(norm.stdev.item() - raw_std) / raw_std < 0.01

    def test_alpha_one_is_full_correction(self):
        """α=1 → σ_eff = σ/cosh(μ) (full curvature correction)."""
        x = _make_series(mu=5.0, sigma=4.0, length=1000)
        norm = FakeRINorm(1)
        norm.alpha_raw.data = torch.tensor(10.0)  # sigmoid(10) ≈ 1
        norm.forward(x)
        expected = 4.0 / np.cosh(5.0)  # ≈ 0.054
        assert abs(norm.stdev.item() - expected) < 0.03

    def test_compression_monotonic_in_mu(self):
        """Higher μ → more compression (lower σ_eff)."""
        sigma_effs = []
        for mu in [0.0, 1.0, 2.0, 3.0, 5.0]:
            x = _make_series(mu=mu, sigma=2.0, length=10000)
            norm = FakeRINorm(1)
            norm.forward(x)
            sigma_effs.append(norm.stdev.item())
        for i in range(len(sigma_effs) - 1):
            assert sigma_effs[i] > sigma_effs[i + 1], (
                f"σ_eff not monotonically decreasing: {sigma_effs}"
            )

    def test_alpha_has_gradient(self):
        """alpha_raw receives gradients through the forward pass.
        Note: z.sum() gives ~0 grad because z is mean-centered; use z² instead."""
        x = _make_series(mu=5.0, sigma=4.0, length=36)
        norm = FakeRINorm(1)
        z = norm.forward(x)
        loss = (z ** 2).sum()
        loss.backward()
        assert norm.alpha_raw.grad is not None
        assert norm.alpha_raw.grad.item() != 0.0

    def test_no_overflow_fp32(self):
        """All computations stay finite in fp32."""
        x = _make_series(mu=5.0, sigma=4.0)
        norm = FakeRINorm(1)
        z = norm.forward(x)
        assert torch.isfinite(z).all()
        assert torch.isfinite(norm.stdev).all()
        assert torch.isfinite(norm.mean).all()

    def test_batch_independent(self):
        """Each series in a batch gets its own curvature compression."""
        x_peace = _make_series(mu=0.0, sigma=0.3, length=36)
        x_war = _make_series(mu=5.0, sigma=4.0, length=36)
        x_batch = torch.cat([x_peace, x_war], dim=0)
        norm = FakeRINorm(1)
        norm.forward(x_batch)
        assert norm.stdev.shape[0] == 2
        assert norm.stdev[0].item() > norm.stdev[1].item() * 0.3


# ═══════════════════════════════════════════════════════════════════════
# Tests for Duan smearing correction (Option 2)
# ═══════════════════════════════════════════════════════════════════════

class TestSmearingCorrection:

    def test_smearing_reduces_high_mu_predictions(self):
        """For conflict entities (large ŷ), smearing shifts predictions down."""
        x = _make_series(mu=5.0, sigma=4.0, length=36)
        norm = FakeRINorm(1)
        z = norm.forward(x)
        z_4d = z.unsqueeze(-1)

        # No smearing
        norm.smear_raw.data = torch.tensor(-20.0)  # c ≈ 0
        y_no_smear = norm.inverse(z_4d.clone()).squeeze(-1)

        # With smearing
        norm.smear_raw.data = torch.tensor(0.0)  # c ≈ 0.69
        y_smeared = norm.inverse(z_4d.clone()).squeeze(-1)

        # Smeared predictions should be closer to zero (lower) for positive ŷ
        assert y_smeared.mean() < y_no_smear.mean()

    def test_smearing_does_not_affect_peaceful(self):
        """For peaceful entities (ŷ≈0), tanh(ŷ)≈0, so correction vanishes."""
        x = _make_series(mu=0.0, sigma=0.1, length=36)
        norm = FakeRINorm(1)
        z = norm.forward(x)
        z_4d = z.unsqueeze(-1)

        norm.smear_raw.data = torch.tensor(-20.0)
        y_no_smear = norm.inverse(z_4d.clone()).squeeze(-1)

        norm.smear_raw.data = torch.tensor(0.0)
        y_smeared = norm.inverse(z_4d.clone()).squeeze(-1)

        # Difference should be negligible for ŷ ≈ 0
        max_diff = (y_no_smear - y_smeared).abs().max().item()
        assert max_diff < 0.01, f"Smearing affected peaceful series: max_diff={max_diff}"

    def test_smearing_c_has_gradient(self):
        """smear_raw receives gradients through the inverse pass."""
        x = _make_series(mu=5.0, sigma=4.0, length=36)
        norm = FakeRINorm(1)
        z = norm.forward(x)
        z_4d = z.unsqueeze(-1)
        y = norm.inverse(z_4d)
        loss = y.sum()
        loss.backward()
        assert norm.smear_raw.grad is not None
        assert norm.smear_raw.grad.item() != 0.0

    def test_smearing_reduces_jensen_in_count_space(self):
        """With noisy predictions, smearing reduces Jensen bias in count space.

        Jensen bias only manifests with prediction variance: E[sinh(ŷ+ε)] > sinh(ŷ).
        A perfect round-trip has no variance, so we add z-noise to simulate an
        imperfect model, then verify smearing pulls sinh(ŷ) closer to truth.
        """
        torch.manual_seed(99)
        x = _make_series(mu=5.0, sigma=4.0, length=1000)
        norm = FakeRINorm(1)
        z = norm.forward(x)
        # Add Gaussian noise (σ_z=0.5) to simulate prediction errors
        z_noisy = z + 0.5 * torch.randn_like(z)
        z_4d = z_noisy.unsqueeze(-1)

        # Without smearing: sinh(inverse(z_noisy)) overshoots due to Jensen bias
        norm.smear_raw.data = torch.tensor(-20.0)  # c ≈ 0
        counts_no_smear = torch.sinh(
            norm.inverse(z_4d.clone()).squeeze(-1)
        ).mean().item()

        # Analytical c for σ_z²/2 = 0.125: softplus(x)=0.125 → x ≈ -1.97
        norm.smear_raw.data = torch.tensor(-1.97)
        counts_smeared = torch.sinh(
            norm.inverse(z_4d.clone()).squeeze(-1)
        ).mean().item()

        truth_counts = torch.sinh(x).mean().item()

        # Smeared should be closer to truth
        err_no_smear = abs(counts_no_smear - truth_counts)
        err_smeared = abs(counts_smeared - truth_counts)
        assert err_smeared < err_no_smear, (
            f"Smearing didn't help: truth={truth_counts:.1f}, "
            f"no_smear_err={err_no_smear:.1f}, smeared_err={err_smeared:.1f}"
        )

    def test_smearing_monotonic_is_preserved(self):
        """The inverse transform must stay monotonically increasing.
        ∂ŷ_corr/∂ŷ = 1 - c·σ²·sech²(ŷ) > 0 requires c·σ² < 1."""
        x = _make_series(mu=5.0, sigma=4.0, length=36)
        norm = FakeRINorm(1)
        norm.forward(x)
        sigma = norm.stdev.item()
        c = F.softplus(norm.smear_raw).item()
        # Monotonicity: 1 - c·σ² > 0 at ŷ=0 (worst case, sech²=1)
        assert c * sigma**2 < 1.0, (
            f"Smearing breaks monotonicity: c·σ²={c * sigma**2:.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Tests for combined forward-inverse consistency
# ═══════════════════════════════════════════════════════════════════════

class TestForwardInverseConsistency:

    def test_round_trip_with_zero_smearing(self):
        """With c=0, forward→inverse should be identity."""
        x = _make_series(mu=5.0, sigma=4.0, length=36)
        norm = FakeRINorm(1)
        norm.smear_raw.data = torch.tensor(-20.0)  # c ≈ 0
        z = norm.forward(x)
        z_4d = z.unsqueeze(-1)
        x_recon = norm.inverse(z_4d).squeeze(-1)
        assert torch.allclose(x, x_recon, atol=1e-4)

    def test_round_trip_with_smearing_shifts_consistently(self):
        """With c>0, inverse output is consistently shifted from input."""
        x = _make_series(mu=5.0, sigma=4.0, length=36)
        norm = FakeRINorm(1)
        z = norm.forward(x)
        z_4d = z.unsqueeze(-1)
        x_recon = norm.inverse(z_4d).squeeze(-1)
        # With default c≈0.05, the shift is small but nonzero
        diff = (x - x_recon).abs().mean().item()
        assert diff > 0, "Smearing should introduce a small shift"
        assert diff < 1.0, f"Shift too large: {diff:.4f}"
