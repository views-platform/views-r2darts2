import pytest
import torch
from views_r2darts2.catalogs.loss_catalog import LossCatalog
from views_r2darts2.catalogs.optimizer_catalog import OptimizerCatalog
from views_r2darts2.catalogs.model_catalog import ModelCatalog

def test_optimizer_catalog_genome_enforcement():
    """Beige Team: Verify OptimizerCatalog genome enforcement."""
    config = {"optimizer_cls": "AdamW", "lr": 0.001, "weight_decay": 0.01}
    opt_cat = OptimizerCatalog(config)
    assert opt_cat.get_optimizer_kwargs() == {"lr": 0.001, "weight_decay": 0.01}
    assert opt_cat.get_optimizer_cls() == torch.optim.AdamW

    config_sgd = {"optimizer_cls": "SGD", "lr": 0.01, "weight_decay": 0.0, "momentum": 0.9}
    assert OptimizerCatalog(config_sgd).get_optimizer_kwargs()["momentum"] == 0.9

    config_sgd_fail = {"optimizer_cls": "SGD", "lr": 0.01, "weight_decay": 0.0}
    with pytest.raises(ValueError, match="MANDATORY OPTIMIZER GENES MISSING"):
        OptimizerCatalog(config_sgd_fail).get_optimizer_kwargs()
    print("✓ Optimizer Catalog Genome Enforcement Verified.")

def test_loss_catalog_genome_enforcement():
    """Beige Team: Verify LossCatalog genome enforcement."""
    config = {
        "loss_function": "WeightedPenaltyHuberLoss",
        "zero_threshold": 0.01, "delta": 0.5, "non_zero_weight": 5.0,
        "false_positive_weight": 2.0, "false_negative_weight": 3.0
    }
    assert LossCatalog(config).get_loss().__class__.__name__ == "WeightedPenaltyHuberLoss"

    config_fail = {"loss_function": "WeightedPenaltyHuberLoss", "delta": 0.5}
    with pytest.raises(ValueError, match="MANDATORY LOSS GENES MISSING"):
        LossCatalog(config_fail).get_loss()
    print("✓ Loss Catalog Genome Enforcement Verified.")

def test_model_catalog_delegation():
    """Green Team: Verify ModelCatalog delegation."""
    config = {
        "algorithm": "NLinearModel", "name": "test_model", "random_state": 42,
        "loss_function": "MSELoss", "optimizer_cls": "Adam", "lr": 0.001, "weight_decay": 0.0,
        "batch_size": 32, "n_epochs": 1, "input_chunk_length": 12, "output_chunk_length": 1,
        "output_chunk_shift": 0, "shared_weights": True, "const_init": True, "normalize": False,
        "use_static_covariates": False, "use_reversible_instance_norm": False,
        "early_stopping_patience": 5, "early_stopping_min_delta": 0.0,
        "lr_scheduler_factor": 0.1, "lr_scheduler_patience": 3, "lr_scheduler_min_lr": 1e-6,
        "gradient_clip_val": 1.0, "steps": [1], "run_type": "test", "num_samples": 1, "mc_dropout": False
    }
    catalog = ModelCatalog(config)
    model = catalog.get_model("NLinearModel")
    assert isinstance(model.model_params["loss_fn"], torch.nn.MSELoss)
    assert model.model_params["optimizer_cls"] == torch.optim.Adam
    print("✓ Model Catalog Delegation Verified.")

if __name__ == "__main__":
    try:
        test_optimizer_catalog_genome_enforcement()
        test_loss_catalog_genome_enforcement()
        test_model_catalog_delegation()
        print("\nALL GENOMIC HANDSHAKE TESTS PASSED. 🖖")
    except Exception as e:
        print(f"\nTEST FAILURE: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
