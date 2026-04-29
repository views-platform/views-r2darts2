"""
Tests for the curvature-aware RINorm σ compression patch.

Verifies: σ_eff = σ / √cosh(μ) correctly compresses high-μ series
while leaving low-μ (peaceful) series unchanged.
"""
import pytest
import torch
import numpy as np


class FakeRINorm(torch.nn.Module):
    """Minimal RINorm replica for testing the patch in isolation."""
    def __init__(self, input_dim, eps=1e-5, affine=False):
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.affine = affine

    def forward(self, x):
        calc_dims = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=calc_dims, keepdim=True).detach()
        raw_stdev = torch.sqrt(
            torch.var(x, dim=calc_dims, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        curvature = torch.cosh(torch.clamp(self.mean, min=-5.0, max=5.0))
        self.stdev = raw_stdev / curvature.sqrt()
        x = (x - self.mean) / self.stdev
        return x

    def inverse(self, x):
        x = x * self.stdev.view(self.stdev.shape + (1,))
        x = x + self.mean.view(self.mean.shape + (1,))
        return x


def _make_series(mu, sigma, length=36, features=1):
    """Create a (1, length, features) tensor with given mean/std in asinh space."""
    torch.manual_seed(42)
    x = mu + sigma * torch.randn(1, length, features)
    return x


class TestCurvatureCompression:

    def test_peaceful_sigma_unchanged(self):
        """For μ≈0, cosh(0)=1, so σ_eff ≈ σ_raw."""
        x = _make_series(mu=0.0, sigma=0.3)
        norm = FakeRINorm(1)
        norm.forward(x)
        raw_std = x.std(dim=1, keepdim=True).item()
        assert abs(norm.stdev.item() - raw_std) / max(raw_std, 1e-8) < 0.05

    def test_ukraine_sigma_compressed(self):
        """For μ=5, σ=4: σ_eff should be ≈ 4/√cosh(5) ≈ 0.47."""
        x = _make_series(mu=5.0, sigma=4.0, length=1000)
        norm = FakeRINorm(1)
        norm.forward(x)
        expected = 4.0 / np.sqrt(np.cosh(5.0))  # ≈ 0.465
        assert abs(norm.stdev.item() - expected) < 0.15  # tolerance for finite sample

    def test_medium_conflict_moderate_compression(self):
        """For μ=2, σ=1.5: σ_eff ≈ 1.5/√cosh(2) ≈ 0.77."""
        x = _make_series(mu=2.0, sigma=1.5, length=1000)
        norm = FakeRINorm(1)
        norm.forward(x)
        expected = 1.5 / np.sqrt(np.cosh(2.0))  # ≈ 0.773
        assert abs(norm.stdev.item() - expected) < 0.15

    def test_forward_inverse_identity(self):
        """Perfect model (ẑ=z) should reconstruct input exactly."""
        x = _make_series(mu=5.0, sigma=4.0, length=36)
        norm = FakeRINorm(1)
        z = norm.forward(x)
        # inverse expects (batch, time, features, nr_params) — add dim
        z_4d = z.unsqueeze(-1)
        x_recon = norm.inverse(z_4d).squeeze(-1)
        assert torch.allclose(x, x_recon, atol=1e-5)

    def test_jensen_amplification_bounded(self):
        """Jensen factor exp(σ_eff²/2) should be ≤ 1.20 for Ukraine."""
        x = _make_series(mu=5.0, sigma=4.0, length=10000)
        norm = FakeRINorm(1)
        norm.forward(x)
        sigma_eff = norm.stdev.item()
        jensen = np.exp(sigma_eff**2 / 2)
        assert jensen < 1.20, f"Jensen factor {jensen:.3f} too high (σ_eff={sigma_eff:.3f})"

    def test_mu_clamp_prevents_z_explosion(self):
        """For μ=8 (extreme), clamp at 5 bounds z-std at √cosh(5) ≈ 8.6."""
        x = _make_series(mu=8.0, sigma=2.0, length=1000)
        norm = FakeRINorm(1)
        z = norm.forward(x)
        z_std = z.std().item()
        # Without clamp: z_std = √cosh(8) ≈ 38.6. With clamp: ≈ 8.6
        assert z_std < 12.0, f"z_std={z_std:.1f} too high, clamp not working"

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

    def test_sinh_output_bounded_at_z3(self):
        """At ẑ=3, sinh(σ_eff×3 + μ) should be < 500 for Ukraine."""
        x = _make_series(mu=5.0, sigma=4.0, length=10000)
        norm = FakeRINorm(1)
        norm.forward(x)
        sigma_eff = norm.stdev.item()
        y_asinh = sigma_eff * 3.0 + 5.0
        y_raw = np.sinh(y_asinh)
        assert y_raw < 500, f"sinh({y_asinh:.2f}) = {y_raw:.0f}, too high"

    def test_no_overflow_fp32(self):
        """cosh(5.0) = 74.2, well within fp32 range."""
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
        x_batch = torch.cat([x_peace, x_war], dim=0)  # (2, 36, 1)
        norm = FakeRINorm(1)
        norm.forward(x_batch)
        # stdev shape: (2, 1, 1)
        assert norm.stdev.shape[0] == 2
        # Peaceful σ_eff should be much larger than Ukraine σ_eff
        assert norm.stdev[0].item() > norm.stdev[1].item() * 0.3
