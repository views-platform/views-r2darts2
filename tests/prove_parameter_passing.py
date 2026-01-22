import unittest
import sys

# Ensure the path is correct to import from the parent directory's module
sys.path.append('..')
from views_r2darts2.utils.loss import LossSelector, AsymmetricQuantileLoss

class TestParameterPassing(unittest.TestCase):

    def test_loss_selector_passes_tau(self):
        """
        Tests if the LossSelector correctly forwards the 'tau' hyperparameter
        to the AsymmetricQuantileLoss constructor.
        """
        print("\n--- Verifying Hyperparameter Pipeline for AsymmetricQuantileLoss ---")

        # --- Case 1: tau = 0.95 ---
        config_1 = {
            'loss_function': 'AsymmetricQuantileLoss',
            'tau': 0.95,
            'lr': 0.001, # Include other irrelevant params to ensure they are ignored
            'batch_size': 128
        }
        print(f"Attempting to create loss with config: {config_1}")
        loss_instance_1 = LossSelector.get_loss_function(config_1['loss_function'], **config_1)

        # Verification
        self.assertIsInstance(loss_instance_1, AsymmetricQuantileLoss, "Failed to create correct loss class.")
        self.assertEqual(loss_instance_1.tau, 0.95, "Failed to pass tau=0.95. The parameter was ignored.")
        print(f"SUCCESS: Loss function created with tau = {loss_instance_1.tau}")

        # --- Case 2: tau = 0.50 ---
        config_2 = {
            'loss_function': 'AsymmetricQuantileLoss',
            'tau': 0.50,
            'weight_decay': 0.0001
        }
        print(f"\nAttempting to create loss with config: {config_2}")
        loss_instance_2 = LossSelector.get_loss_function(config_2['loss_function'], **config_2)
        
        # Verification
        self.assertIsInstance(loss_instance_2, AsymmetricQuantileLoss, "Failed to create correct loss class.")
        self.assertEqual(loss_instance_2.tau, 0.50, "Failed to pass tau=0.50. The parameter was ignored.")
        print(f"SUCCESS: Loss function created with tau = {loss_instance_2.tau}")

        # --- Case 3: Using default tau ---
        config_3 = {
            'loss_function': 'AsymmetricQuantileLoss',
            # 'tau' is deliberately omitted
        }
        print(f"\nAttempting to create loss with config: {config_3}")
        loss_instance_3 = LossSelector.get_loss_function(config_3['loss_function'], **config_3)
        
        # Verification
        # The default tau in the class definition is 0.75
        self.assertIsInstance(loss_instance_3, AsymmetricQuantileLoss, "Failed to create correct loss class.")
        self.assertEqual(loss_instance_3.tau, 0.75, "Loss function did not fall back to the default tau=0.75.")
        print(f"SUCCESS: Loss function correctly fell back to default tau = {loss_instance_3.tau}")
        
        print("\n--- VERDICT: The hyperparameter pipeline via LossSelector is CORRECT. ---")


if __name__ == '__main__':
    # We need to run this from the 'tests' directory for the path to work
    # Or run with `python -m tests.prove_parameter_passing` from the root
    unittest.main()
