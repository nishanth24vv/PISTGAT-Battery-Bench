"""Loss functions for physics-informed battery health forecasting."""

from pistgat_bench.losses.monotonicity import CapacityMonotonicityLoss, PhysicsInformedLoss

__all__ = ["CapacityMonotonicityLoss", "PhysicsInformedLoss"]
