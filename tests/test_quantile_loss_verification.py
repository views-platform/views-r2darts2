import torch
import unittest
from torch.autograd import gradcheck
import sys
# Ensure the path is correct to import from the parent directory's module
sys.path.append('..')
from views_r2darts2.utils.loss import AsymmetricQuantileLoss

class TestQuantileLossVerification(unittest.TestCase):

    def test_golden_values(self):
        """Numerically prove correctness against hand-calculated 'golden' values."""
        
        # --- Test Case 1: Underprediction (error > 0) ---
        loss_fn_75 = AsymmetricQuantileLoss(tau=0.75, non_zero_weight=1.0) # Disable weighting
        preds = torch.tensor([1.0, 2.0])
        targets = torch.tensor([3.0, 3.0])
        # Expected loss: (0.75 * 2.0 + 0.75 * 1.0) / 2 = (1.5 + 0.75) / 2 = 1.125
        self.assertAlmostEqual(loss_fn_75(preds, targets).item(), 1.125, places=6)

        # --- Test Case 2: Overprediction (error < 0) ---
        preds = torch.tensor([4.0, 4.0])
        targets = torch.tensor([2.0, 3.0])
        # Expected loss: ((1-0.75) * |-2.0| + (1-0.75) * |-1.0|) / 2 = (0.25 * 2.0 + 0.25 * 1.0) / 2 = (0.5 + 0.25) / 2 = 0.375
        self.assertAlmostEqual(loss_fn_75(preds, targets).item(), 0.375, places=6)

        # --- Test Case 3: Perfect prediction ---
        preds = torch.tensor([5.0, 5.0])
        targets = torch.tensor([5.0, 5.0])
        # Expected loss: 0
        self.assertAlmostEqual(loss_fn_75(preds, targets).item(), 0.0, places=6)

        # --- Test Case 4: tau=0.5 should be 0.5 * MAE ---
        loss_fn_50 = AsymmetricQuantileLoss(tau=0.5, non_zero_weight=1.0)
        preds = torch.tensor([1.0, 4.0])
        targets = torch.tensor([3.0, 1.0])
        # Expected loss: 0.5 * MAE = 1.25
        self.assertAlmostEqual(loss_fn_50(preds, targets).item(), 1.25, places=6)

        # --- Test Case 5: non_zero_weight application ---
        loss_fn_weighted = AsymmetricQuantileLoss(tau=0.8, non_zero_weight=10.0, zero_threshold=0.5)
        preds = torch.tensor([1.0, 1.0])
        targets = torch.tensor([0.0, 2.0]) # One zero, one non-zero
        # Loss for target=0.0 (overprediction, weight=1.0): (1-0.8) * |-1.0| * 1.0 = 0.2
        # Loss for target=2.0 (underprediction, weight=10.0): 0.8 * 1.0 * 10.0 = 8.0
        # Expected mean loss: (0.2 + 8.0) / 2 = 4.1
        self.assertAlmostEqual(loss_fn_weighted(preds, targets).item(), 4.1, places=6)
        print("\nGolden vector tests passed.")

    def test_gradient_correctness(self):
        """Prove gradient correctness with torch.autograd.gradcheck."""
        
        # Use double precision for gradcheck
        loss_fn = AsymmetricQuantileLoss(tau=0.8, non_zero_weight=10.0).double()
        
        # Inputs must be double and require grad.
        # Ensure error is non-zero to avoid the non-differentiable point.
        preds = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        targets = torch.randn(5, 5, dtype=torch.double)
        
        # Ensure no zero errors
        targets[torch.abs(preds - targets) < 1e-3] += 0.1
        
        # gradcheck takes a function and a tuple of inputs
        gradcheck_passed = gradcheck(loss_fn, (preds, targets), eps=1e-6, atol=1e-4)
        self.assertTrue(gradcheck_passed)
        print("Gradient correctness check passed.")

if __name__ == '__main__':
    unittest.main()
