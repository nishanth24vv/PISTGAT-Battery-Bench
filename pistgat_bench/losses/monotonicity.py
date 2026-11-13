"""Physics-Informed Capacity Monotonicity Loss for battery SOH degradation."""

from __future__ import annotations

import torch
import torch.nn as nn


class CapacityMonotonicityLoss(nn.Module):
    """Penalizes non-physical capacity recoveries in predicted SOH sequences.

    During battery aging, capacity degrades monotonically except for minor relaxation
    effects during rest periods. This loss heavily penalizes consecutive cycle predictions
    where SOH(t+1) > SOH(t) + epsilon_regen.
    """

    def __init__(self, epsilon_regen: float = 0.005, penalty_weight: float = 1.0):
        super().__init__()
        self.epsilon_regen = epsilon_regen
        self.penalty_weight = penalty_weight

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate monotonicity violation penalty.

        Args:
            y_pred: Tensor of predicted SOH values. Can be:
                    - 1D sequence (T,)
                    - 2D batch of sequences (B, T) or (T, num_nodes)
                    - 2D window batch where ordering is temporal along dim 0 or 1.

        Returns:
            Scalar penalty loss.
        """
        if y_pred.numel() <= 1:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        # Ensure we compute diff along temporal dimension
        if y_pred.ndim == 1:
            diff = y_pred[1:] - y_pred[:-1]
        elif y_pred.ndim == 2:
            # Assume time is along dimension 0 if dim 0 > dim 1, else along dimension 1
            if y_pred.size(0) >= y_pred.size(1):
                diff = y_pred[1:, :] - y_pred[:-1, :]
            else:
                diff = y_pred[:, 1:] - y_pred[:, :-1]
        else:
            diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]

        # Violations occur when positive difference exceeds allowed regeneration epsilon
        violations = torch.relu(diff - self.epsilon_regen)
        loss = torch.mean(torch.square(violations))
        return self.penalty_weight * loss


class PhysicsInformedLoss(nn.Module):
    """Composite loss combining MSE loss with Physics-Guided Capacity Monotonicity."""

    def __init__(self, lambda_mono: float = 0.5, epsilon_regen: float = 0.005):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mono_loss = CapacityMonotonicityLoss(epsilon_regen=epsilon_regen, penalty_weight=lambda_mono)
        self.lambda_mono = lambda_mono

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute composite loss."""
        mse_val = self.mse(y_pred.view(-1), y_true.view(-1))
        mono_val = self.mono_loss(y_pred)
        return mse_val + mono_val
