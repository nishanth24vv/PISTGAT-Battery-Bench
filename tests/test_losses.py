"""Unit tests for Physics-Informed Capacity Monotonicity Loss."""

import pytest
import torch

from pistgat_bench.losses.monotonicity import CapacityMonotonicityLoss, PhysicsInformedLoss


def test_monotonicity_loss_decreasing_sequence():
    """A strictly degrading sequence should incur near-zero monotonicity loss."""
    loss_fn = CapacityMonotonicityLoss(epsilon_regen=0.005)

    # Monotonically decaying SOH from 1.0 down to 0.70
    decaying = torch.linspace(1.0, 0.70, steps=50)
    penalty = loss_fn(decaying)
    assert penalty.item() == 0.0


def test_monotonicity_loss_increasing_sequence():
    """A sequence with non-physical capacity jumps should incur positive penalty."""
    loss_fn = CapacityMonotonicityLoss(epsilon_regen=0.005, penalty_weight=1.0)

    # Sequence with sharp unphysical increases
    increasing = torch.linspace(0.70, 0.95, steps=50)
    penalty = loss_fn(increasing)
    assert penalty.item() > 0.0


def test_physics_informed_composite_loss_backward():
    """Ensure gradient backpropagation works through the composite physics loss."""
    loss_fn = PhysicsInformedLoss(lambda_mono=0.5, epsilon_regen=0.005)

    y_pred = torch.tensor([0.90, 0.92, 0.94, 0.88], requires_grad=True)
    y_true = torch.tensor([0.90, 0.88, 0.86, 0.84])

    loss = loss_fn(y_pred, y_true)
    assert loss.item() > 0.0

    loss.backward()
    assert y_pred.grad is not None
    assert torch.all(torch.isfinite(y_pred.grad))
