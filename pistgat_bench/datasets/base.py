"""Abstract base class and cycle data representations for battery datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass
class BatteryCycleRecord:
    """Represents standardized telemetry from a single battery charge/discharge cycle."""
    cycle_index: int
    capacity_ah: float
    soh: float  # Capacity / Nominal Capacity in [0.0, 1.0]
    internal_resistance_ohm: float = 0.08
    voltage_mean: float = 3.7
    voltage_min: float = 2.7
    voltage_max: float = 4.2
    current_mean: float = -2.0
    temp_max: float = 25.0
    duration_s: float = 3600.0
    voltage_curve: List[float] = field(default_factory=list)
    current_curve: List[float] = field(default_factory=list)
    temp_curve: List[float] = field(default_factory=list)

    def to_feature_vector(self) -> np.ndarray:
        """Standard 6-dimensional summary feature vector for the cycle."""
        return np.array([
            self.voltage_mean,
            self.voltage_min,
            self.voltage_max,
            self.current_mean,
            self.temp_max,
            self.internal_resistance_ohm
        ], dtype=np.float32)


class BaseBatteryDataset(ABC):
    """Unified abstract battery dataset interface for benchmark baselines and PISTGAT."""

    def __init__(self, name: str, nominal_capacity_ah: float):
        self.name = name
        self.nominal_capacity_ah = nominal_capacity_ah
        self.cell_data: Dict[str, List[BatteryCycleRecord]] = {}
        self.is_loaded = False
        # Normalization scale caches
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    @abstractmethod
    def load(self, source_path: Optional[str] = None) -> BaseBatteryDataset:
        """Load and parse dataset into self.cell_data."""
        pass

    def get_cell_ids(self) -> List[str]:
        """Return list of available cell identifiers."""
        return sorted(list(self.cell_data.keys()))

    def get_cell_cycles(self, cell_id: str) -> List[BatteryCycleRecord]:
        """Retrieve full cycle telemetry series for specified cell."""
        if cell_id not in self.cell_data:
            raise KeyError(f"Cell '{cell_id}' not found in dataset {self.name}. Available: {self.get_cell_ids()}")
        return self.cell_data[cell_id]

    def get_features_and_targets(
        self,
        cell_id: str,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract continuous feature matrix (T, D) and target SOH vector (T,)."""
        cycles = self.get_cell_cycles(cell_id)
        X = np.stack([c.to_feature_vector() for c in cycles], axis=0)
        y = np.array([c.soh for c in cycles], dtype=np.float32)

        if normalize:
            if self._feature_means is None or self._feature_stds is None:
                self._compute_normalization_stats()
            X = (X - self._feature_means) / (self._feature_stds + 1e-6)

        return X, y

    def _compute_normalization_stats(self) -> None:
        """Compute dataset-wide statistics across all loaded cells."""
        all_features = []
        for cycles in self.cell_data.values():
            for c in cycles:
                all_features.append(c.to_feature_vector())
        if not all_features:
            self._feature_means = np.zeros(6, dtype=np.float32)
            self._feature_stds = np.ones(6, dtype=np.float32)
            return
        stacked = np.stack(all_features, axis=0)
        self._feature_means = np.mean(stacked, axis=0)
        self._feature_stds = np.std(stacked, axis=0)

    def create_sliding_windows(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window_size: int = 10,
        stride: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert sequence into sliding temporal windows for sequence models (LSTM / Transformer)."""
        num_cycles = len(X)
        if num_cycles < window_size:
            raise ValueError(f"Number of cycles ({num_cycles}) is smaller than window_size ({window_size})")

        windows_X = []
        targets_y = []
        for i in range(0, num_cycles - window_size + 1, stride):
            windows_X.append(X[i : i + window_size])
            targets_y.append(y[i + window_size - 1])

        return (
            torch.tensor(np.array(windows_X), dtype=torch.float32),
            torch.tensor(np.array(targets_y), dtype=torch.float32)
        )

    def get_spatiotemporal_graph(
        self,
        cell_ids: Optional[List[str]] = None,
        window_size: int = 10,
        stride: int = 1,
        pack_topology: str = "series"
    ) -> List[Data]:
        """Construct PyTorch Geometric Data graphs representing pack topology over time.

        Each node represents a cell with its temporal feature window (window_size, D).
        Edges represent electrical series connections and adjacent thermal transfer paths.
        """
        if cell_ids is None:
            cell_ids = self.get_cell_ids()

        num_cells = len(cell_ids)
        if num_cells < 2:
            # Need at least two cells for graph connectivity; can duplicate single cell if testing
            cell_ids = cell_ids * 2 if cell_ids else ["Cell_1", "Cell_2"]
            num_cells = len(cell_ids)

        # Collect synchronized windowed sequences
        cell_windows = []
        cell_targets = []
        min_windows = float("inf")

        for cid in cell_ids:
            X, y = self.get_features_and_targets(cid, normalize=True)
            w_X, w_y = self.create_sliding_windows(X, y, window_size=window_size, stride=stride)
            cell_windows.append(w_X)
            cell_targets.append(w_y)
            min_windows = min(min_windows, len(w_X))

        # Build bidirectional edges: adjacent cells in string (i <-> i+1)
        edge_sources = []
        edge_targets = []
        for i in range(num_cells - 1):
            edge_sources.extend([i, i + 1])
            edge_targets.extend([i + 1, i])
        
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

        # Build list of graph samples for each time window
        graph_samples: List[Data] = []
        for t in range(int(min_windows)):
            # x shape: (num_cells, window_size, feature_dim)
            x_nodes = torch.stack([cell_windows[c][t] for c in range(num_cells)], dim=0)
            # y shape: (num_cells,) - SOH targets for all cells at this timestep
            y_nodes = torch.stack([cell_targets[c][t] for c in range(num_cells)], dim=0)

            graph_data = Data(
                x=x_nodes,
                edge_index=edge_index,
                y=y_nodes,
                num_nodes=num_cells
            )
            graph_samples.append(graph_data)

        return graph_samples

    def leave_one_cell_out_split(
        self,
        test_cell_id: str
    ) -> Tuple[List[str], List[str]]:
        """Split available cell IDs for Leave-One-Cell-Out (LOCO) cross-validation."""
        all_cells = self.get_cell_ids()
        if test_cell_id not in all_cells:
            raise ValueError(f"Test cell '{test_cell_id}' not found in {all_cells}")
        train_cells = [c for c in all_cells if c != test_cell_id]
        return train_cells, [test_cell_id]

    def temporal_split(
        self,
        cell_id: str,
        train_ratio: float = 0.7
    ) -> Tuple[List[BatteryCycleRecord], List[BatteryCycleRecord]]:
        """Split a single cell degradation history into early train and late test cycles."""
        cycles = self.get_cell_cycles(cell_id)
        split_idx = int(len(cycles) * train_ratio)
        return cycles[:split_idx], cycles[split_idx:]
