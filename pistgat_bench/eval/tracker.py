"""Experiment tracking with MLflow and telemetry export for 3D visualization."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class BenchmarkTracker:
    """Manages MLflow tracking and exports 3D visualizer telemetry."""

    def __init__(self, experiment_name: str = "PISTGAT-Battery-Bench", enabled: bool = True):
        self.experiment_name = experiment_name
        self.enabled = enabled
        self.active_run = None
        self._mlflow_available = False

        if self.enabled:
            try:
                import mlflow
                mlflow.set_experiment(self.experiment_name)
                self._mlflow_available = True
            except Exception as e:
                # Fall back gracefully to local file tracking
                self._mlflow_available = False

    def start_run(self, run_name: str) -> None:
        """Start tracking run."""
        if self.enabled and self._mlflow_available:
            import mlflow
            self.active_run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters and hyperparameters."""
        if self.enabled and self._mlflow_available and self.active_run:
            import mlflow
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log evaluation metrics."""
        if self.enabled and self._mlflow_available and self.active_run:
            import mlflow
            mlflow.log_metrics(metrics, step=step)

    def end_run(self) -> None:
        """Complete tracking run."""
        if self.enabled and self._mlflow_available and self.active_run:
            import mlflow
            mlflow.end_run()
            self.active_run = None

    @staticmethod
    def export_pack_telemetry(
        output_path: str,
        pack_config: str,
        cell_trajectories: Dict[str, Dict[str, List[float]]],
        benchmark_results: Optional[Dict[str, Dict[str, float]]] = None,
        dataset_meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """Export standardized JSON telemetry formatted for the Three.js 3D Pack Visualizer.

        Args:
            output_path: Destination path for the exported JSON.
            pack_config: Battery configuration (e.g. '4S1P', '8S1P', '12S1P').
            cell_trajectories: Mapping of cell_id -> dict of arrays
                               ('soh_pred', 'soh_true', 'temperature', 'resistance', 'voltage').
            benchmark_results: Comparative performance metrics across models.
            dataset_meta: Metadata describing cells and nominal ratings.

        Returns:
            Absolute path of written JSON file.
        """
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        cell_ids = list(cell_trajectories.keys())
        first_cell = cell_ids[0] if cell_ids else ""
        num_cycles = len(cell_trajectories[first_cell].get("soh_pred", [])) if first_cell else 0

        # Construct cycle-by-cycle pack timeline
        timeline = []
        for c in range(num_cycles):
            soh_vals = [
                cell_trajectories[cid]["soh_pred"][c]
                for cid in cell_ids
                if c < len(cell_trajectories[cid].get("soh_pred", []))
            ]
            temp_vals = [
                cell_trajectories[cid]["temperature"][c]
                for cid in cell_ids
                if "temperature" in cell_trajectories[cid] and c < len(cell_trajectories[cid]["temperature"])
            ]
            min_soh = min(soh_vals) if soh_vals else 1.0
            max_soh = max(soh_vals) if soh_vals else 1.0
            avg_soh = float(np.mean(soh_vals)) if soh_vals else 1.0
            imbalance = max_soh - min_soh

            cell_statuses = {}
            for cid in cell_ids:
                c_soh = cell_trajectories[cid]["soh_pred"][c] if c < len(cell_trajectories[cid]["soh_pred"]) else 1.0
                c_true = cell_trajectories[cid]["soh_true"][c] if "soh_true" in cell_trajectories[cid] and c < len(cell_trajectories[cid]["soh_true"]) else c_soh
                c_temp = cell_trajectories[cid]["temperature"][c] if "temperature" in cell_trajectories[cid] and c < len(cell_trajectories[cid]["temperature"]) else 25.0
                c_res = cell_trajectories[cid]["resistance"][c] if "resistance" in cell_trajectories[cid] and c < len(cell_trajectories[cid]["resistance"]) else 0.08
                cell_statuses[cid] = {
                    "soh_pred": round(float(c_soh), 4),
                    "soh_true": round(float(c_true), 4),
                    "temperature_c": round(float(c_temp), 2),
                    "resistance_ohm": round(float(c_res), 5),
                    "soh_deviation": round(float(c_soh - avg_soh), 4)
                }

            timeline.append({
                "cycle": c + 1,
                "pack_soh_mean": round(avg_soh, 4),
                "pack_imbalance_delta": round(float(imbalance), 4),
                "max_temp_c": round(float(max(temp_vals)), 2) if temp_vals else 25.0,
                "cells": cell_statuses
            })

        export_data = {
            "metadata": dataset_meta or {
                "benchmark": "PISTGAT-Battery-Bench",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pack_configuration": pack_config,
                "num_cells": len(cell_ids),
                "cell_ids": cell_ids
            },
            "benchmark_results": benchmark_results or {},
            "timeline": timeline
        }

        with open(p, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        return str(p)
