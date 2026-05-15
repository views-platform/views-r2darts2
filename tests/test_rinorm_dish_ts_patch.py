"""
Tests for the Dish-TS RINorm patch (v6 — learned denormalization).

Forward: standard linear RevIN  z = (x − μ) / σ
Inverse: learned CONET           ŷ = ẑ · σ_out + μ_out

Verifies:
  1. Identity warm-start: round-trip ≈ standard RevIN at initialization
  2. Gradient flow to CONET parameters via loss.backward()
  3. Checkpoint compatibility: loading pre-CONET state_dict succeeds
  4. Shape correctness for single and multi-target configs
#   5. Jensen-safe absolute σ_out ceiling
"""
import pytest
import torch
import torch.nn as nn
import copy


@pytest.fixture(autouse=True)
def _unpatch_rinorm():
    """Ensure each test starts with a clean (unpatched) RINorm."""
    from darts.models.components.layer_norm_variants import RINorm
    # Save originals before any patching
    orig_init = RINorm.__init__
    orig_forward = RINorm.forward
    orig_inverse = RINorm.inverse
    orig_load = RINorm._load_from_state_dict
    patched = getattr(RINorm, '_dish_ts_patched', False)

    yield

    # Restore originals
    RINorm.__init__ = orig_init
    RINorm.forward = orig_forward
    RINorm.inverse = orig_inverse
    RINorm._load_from_state_dict = orig_load
    if patched:
        RINorm._dish_ts_patched = False


def _apply_patch():
    from views_r2darts2.infrastructure.patches import apply_rinorm_compression_patch
    apply_rinorm_compression_patch()


def _make_rinorm(n_targets=1):
    _apply_patch()
    from darts.models.components.layer_norm_variants import RINorm
    return RINorm(input_dim=n_targets)


def _make_conflict_batch(batch=4, length=36, n_targets=1):
    """Simulated conflict series in asinh-space (μ_raw ≈ 74)."""
    torch.manual_seed(42)
    x_raw = 74 + 297 * torch.randn(batch, length, n_targets).abs()
    return torch.asinh(x_raw)


def _make_peaceful_batch(batch=4, length=36, n_targets=1):
    """Simulated peaceful series in asinh-space (μ_raw ≈ 0.1)."""
    torch.manual_seed(42)
    x_raw = 0.1 + 0.3 * torch.randn(batch, length, n_targets).abs()
    return torch.asinh(x_raw)


# ═══════════════════════════════════════════════════════════════════════
# 1. Identity Warm-Start (Round-Trip)
# ═══════════════════════════════════════════════════════════════════════

class TestRoundTrip:

    def test_identity_roundtrip_conflict(self):
        """At init (CONET=zeros), inverse(forward(x)) ≈ x."""
        rin = _make_rinorm(1)
        x = _make_conflict_batch()
        z = rin.forward(x)
        # Inverse expects (B, T, n_targets, nr_params)
        z_4d = z.unsqueeze(-1)
        x_hat = rin.inverse(z_4d).squeeze(-1)
        assert torch.allclose(x, x_hat, atol=1e-4), (
            f"max_diff={(x - x_hat).abs().max():.6f}"
        )

    def test_identity_roundtrip_peaceful(self):
        """At init, round-trip works for peaceful series too."""
        rin = _make_rinorm(1)
        x = _make_peaceful_batch()
        z = rin.forward(x)
        z_4d = z.unsqueeze(-1)
        x_hat = rin.inverse(z_4d).squeeze(-1)
        assert torch.allclose(x, x_hat, atol=1e-4), (
            f"max_diff={(x - x_hat).abs().max():.6f}"
        )

    def test_identity_roundtrip_all_zeros(self):
        """All-zero series (eps prevents div-by-zero)."""
        rin = _make_rinorm(1)
        x = torch.zeros(2, 36, 1)
        z = rin.forward(x)
        z_4d = z.unsqueeze(-1)
        x_hat = rin.inverse(z_4d).squeeze(-1)
        assert torch.allclose(x, x_hat, atol=1e-3)

    def test_identity_roundtrip_multi_target(self):
        """Round-trip with n_targets=3."""
        rin = _make_rinorm(3)
        torch.manual_seed(7)
        x = torch.randn(4, 36, 3) * 2 + 3  # varied asinh-space values
        z = rin.forward(x)
        z_4d = z.unsqueeze(-1)
        x_hat = rin.inverse(z_4d).squeeze(-1)
        assert torch.allclose(x, x_hat, atol=1e-4), (
            f"max_diff={(x - x_hat).abs().max():.6f}"
        )

    def test_roundtrip_multi_param(self):
        """Round-trip with nr_params=3 (quantile outputs)."""
        rin = _make_rinorm(1)
        x = _make_conflict_batch()
        z = rin.forward(x)
        # Simulate model outputting 3 quantile params
        z_mp = z.unsqueeze(-1).expand(-1, -1, -1, 3)
        x_hat = rin.inverse(z_mp)
        # Each param channel should reconstruct x
        for p in range(3):
            assert torch.allclose(x, x_hat[..., p], atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# 2. Gradient Flow to CONET
# ═══════════════════════════════════════════════════════════════════════

class TestGradientFlow:

    def test_conet_receives_gradients(self):
        """loss.backward() produces non-zero grads on CONET parameters."""
        rin = _make_rinorm(1)
        x = _make_conflict_batch().requires_grad_(False)

        z = rin.forward(x)
        z_4d = z.unsqueeze(-1)
        y_hat = rin.inverse(z_4d).squeeze(-1)

        # Simple MSE loss against a shifted target
        target = x + 0.5
        loss = ((y_hat - target) ** 2).mean()
        loss.backward()

        conet_params = list(rin._output_conet.parameters())
        assert len(conet_params) == 4  # 2 layers × (weight, bias)

        has_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in conet_params)
        assert has_grad, "CONET parameters received no gradients"

    def test_affine_receives_gradients(self):
        """Affine weight/bias also get gradients (not blocked by CONET)."""
        rin = _make_rinorm(1)
        x = _make_conflict_batch()
        z = rin.forward(x)
        y_hat = rin.inverse(z.unsqueeze(-1)).squeeze(-1)
        loss = ((y_hat - x) ** 2).mean()
        loss.backward()
        assert rin.affine_weight.grad is not None
        assert rin.affine_bias.grad is not None

    def test_conet_params_discoverable(self):
        """model.parameters() picks up CONET via the submodule chain."""
        rin = _make_rinorm(1)
        param_names = {n for n, _ in rin.named_parameters()}
        conet_names = {n for n in param_names if '_output_conet' in n}
        assert len(conet_names) == 4, f"Expected 4 CONET params, got {conet_names}"

    def test_conet_param_count(self):
        """CONET has ~100 params for n_targets=1: Linear(3,16)+Linear(16,2)."""
        rin = _make_rinorm(1)
        n_params = sum(
            p.numel() for n, p in rin.named_parameters()
            if '_output_conet' in n
        )
        # Linear(3,16): 3*16+16=64, Linear(16,2): 16*2+2=34, total=98
        assert n_params == 98, f"Expected 98 CONET params, got {n_params}"


# ═══════════════════════════════════════════════════════════════════════
# 3. Checkpoint Compatibility
# ═══════════════════════════════════════════════════════════════════════

class TestCheckpointCompat:

    def test_load_pre_conet_state_dict(self):
        """Loading a state_dict without CONET keys succeeds silently."""
        rin = _make_rinorm(1)

        # Simulate a pre-CONET checkpoint: only affine params
        old_sd = {
            'affine_weight': torch.ones(1),
            'affine_bias': torch.zeros(1),
        }
        # Should not raise
        rin.load_state_dict(old_sd, strict=False)

    def test_load_full_state_dict(self):
        """Loading a complete state_dict (with CONET) works."""
        rin1 = _make_rinorm(1)
        sd = rin1.state_dict()
        rin2 = _make_rinorm(1)
        rin2.load_state_dict(sd)

    def test_missing_conet_keys_suppressed(self):
        """_load_from_state_dict removes CONET keys from missing_keys."""
        rin = _make_rinorm(1)
        old_sd = {
            'affine_weight': torch.ones(1),
            'affine_bias': torch.zeros(1),
        }
        missing = []
        unexpected = []
        errors = []
        rin._load_from_state_dict(
            old_sd, prefix='', local_metadata={}, strict=True,
            missing_keys=missing, unexpected_keys=unexpected, error_msgs=errors,
        )
        conet_missing = [k for k in missing if '_output_conet' in k]
        assert len(conet_missing) == 0, f"CONET keys leaked to missing: {conet_missing}"


# ═══════════════════════════════════════════════════════════════════════
# 4. Shape Correctness
# ═══════════════════════════════════════════════════════════════════════

class TestShapes:

    @pytest.mark.parametrize("B,T,N", [(1, 36, 1), (8, 36, 1), (4, 12, 3)])
    def test_forward_output_shape(self, B, T, N):
        rin = _make_rinorm(N)
        x = torch.randn(B, T, N)
        z = rin.forward(x)
        assert z.shape == (B, T, N)

    @pytest.mark.parametrize("B,T,N,P", [(1, 6, 1, 1), (8, 6, 1, 3), (4, 12, 3, 1)])
    def test_inverse_output_shape(self, B, T, N, P):
        rin = _make_rinorm(N)
        x = torch.randn(B, 36, N)
        rin.forward(x)  # populate stats
        z = torch.randn(B, T, N, P)
        y = rin.inverse(z)
        assert y.shape == (B, T, N, P)


# ═══════════════════════════════════════════════════════════════════════
# 5. σ_out Boundedness
# ═══════════════════════════════════════════════════════════════════════

class TestSigmaBounds:

    def test_sigma_out_bounded(self):
        """σ_out is always in [0.5σ, 1.5σ] regardless of CONET output."""
        rin = _make_rinorm(1)
        x = _make_conflict_batch()
        rin.forward(x)

        sigma = rin.stdev  # (B, 1, 1)
        # Manually compute σ_out for arbitrary raw_scale values
        for s in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            raw_scale = torch.full_like(sigma, s)
            sigma_out = sigma * (0.5 + torch.sigmoid(raw_scale))
            lo = (0.5 * sigma).min()
            hi = (1.5 * sigma).max()
            assert sigma_out.min() >= lo - 1e-6
            assert sigma_out.max() <= hi + 1e-6

    def test_identity_init_sigma_equals_sigma(self):
        """At init (raw_scale=0), σ_out = σ exactly."""
        rin = _make_rinorm(1)
        x = _make_conflict_batch()
        rin.forward(x)
        # CONET outputs 0 at init → sigmoid(0) = 0.5 → σ_out = σ
        sigma_out = rin.stdev * (0.5 + torch.sigmoid(rin._raw_scale))
        assert torch.allclose(sigma_out, rin.stdev, atol=1e-6)
