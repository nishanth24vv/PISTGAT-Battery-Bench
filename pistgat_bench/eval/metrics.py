"""Standardized battery benchmark evaluation metrics."""

from __future__ import annotations

from typing import Dict, Any, Union
import numpy as np


def compute_bench_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    inference_latency_ms: float = 0.0
) -> Dict[str, float]:
    """Compute standard Li-ion SOH benchmark metrics.

    Calculates:
      - RMSE (%): Root Mean Squared Error scaled as percentage
      - MAE (%): Mean Absolute Error scaled as percentage
      - R2: Coefficient of Determination (1.0 is perfect)
      - Max_Error (%): Maximum absolute deviation
      - MAPE (%): Mean Absolute Percentage Error
      - Latency (ms): Average inference time per sample

    Args:
        y_true: Ground truth SOH values in [0, 1].
        y_pred: Predicted SOH values in [0, 1].
        inference_latency_ms: Milliseconds per inference step.

    Returns:
        Dictionary of formatted benchmark metrics.
    """
    yt = np.asarray(y_true, dtype=np.float64).flatten()
    yp = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(yt) == 0 or len(yp) == 0:
        return {
            "rmse_percent": 0.0,
            "mae_percent": 0.0,
            "r2_score": 0.0,
            "max_error_percent": 0.0,
            "mape_percent": 0.0,
            "inference_latency_ms": inference_latency_ms
        }

    errors = yp - yt
    abs_errors = np.abs(errors)

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(abs_errors))
    max_err = float(np.max(abs_errors))

    # Avoid zero division in MAPE
    non_zero = yt > 1e-4
    if np.any(non_zero):
        mape = float(np.mean(abs_errors[non_zero] / yt[non_zero])) * 100.0
    else:
        mape = 0.0

    # R2 Score calculation
    ss_res = np.sum((yt - yp)**2)
    ss_tot = np.sum((yt - np.mean(yt))**2)
    if ss_tot > 1e-7:
        r2 = float(1.0 - (ss_res / ss_tot))
    else:
        r2 = 1.0 if ss_res < 1e-7 else 0.0

    return {
        "rmse_percent": round(rmse * 100.0, 3),
        "mae_percent": round(mae * 100.0, 3),
        "r2_score": round(r2, 4),
        "max_error_percent": round(max_err * 100.0, 3),
        "mape_percent": round(mape, 3),
        "inference_latency_ms": round(inference_latency_ms, 2)
    }
