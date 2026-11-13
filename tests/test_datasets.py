"""Unit tests for battery dataset loaders and spatiotemporal graph constructors."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from pistgat_bench.datasets import get_dataset
from pistgat_bench.datasets.nasa import NASABatteryDataset
from pistgat_bench.datasets.calce import CALCEBatteryDataset
from pistgat_bench.datasets.oxford import OxfordBatteryDataset


def test_nasa_dataset_loading():
    ds = NASABatteryDataset()
    ds.load()
    cells = ds.get_cell_ids()
    assert len(cells) >= 1
    assert "B0005" in cells or "Cell_1" in cells[0]

    cycles = ds.get_cell_cycles(cells[0])
    assert len(cycles) > 10
    assert 0.0 <= cycles[0].soh <= 1.05


def test_calce_dataset_loading():
    ds = CALCEBatteryDataset()
    ds.load()
    cells = ds.get_cell_ids()
    assert len(cells) >= 1
    cycles = ds.get_cell_cycles(cells[0])
    assert len(cycles) > 20
    assert cycles[0].soh >= cycles[-1].soh  # degradation occurs


def test_oxford_dataset_loading():
    ds = OxfordBatteryDataset()
    ds.load()
    cells = ds.get_cell_ids()
    assert len(cells) == 8
    assert "Cell_1" in cells
    cycles = ds.get_cell_cycles("Cell_1")
    assert len(cycles) > 20


def test_sliding_windows_shapes():
    ds = get_dataset("nasa").load()
    cell_id = ds.get_cell_ids()[0]
    X, y = ds.get_features_and_targets(cell_id, normalize=True)

    window_size = 8
    X_win, y_win = ds.create_sliding_windows(X, y, window_size=window_size, stride=1)

    assert isinstance(X_win, torch.Tensor)
    assert isinstance(y_win, torch.Tensor)
    assert X_win.ndim == 3
    assert X_win.shape[1] == window_size
    assert X_win.shape[2] == 6  # 6 features
    assert len(X_win) == len(y_win)
    assert len(X_win) == len(X) - window_size + 1


def test_spatiotemporal_graph_construction():
    ds = get_dataset("nasa").load()
    cell_ids = ds.get_cell_ids()[:4]
    graphs = ds.get_spatiotemporal_graph(cell_ids=cell_ids, window_size=6)

    assert len(graphs) > 0
    g = graphs[0]
    assert isinstance(g, Data)
    assert g.num_nodes == len(cell_ids)
    assert g.x.shape == (len(cell_ids), 6, 6)  # (nodes, window_size, feature_dim)
    assert g.edge_index.shape[0] == 2
    assert g.y.shape == (len(cell_ids),)


def test_loco_split():
    ds = get_dataset("nasa").load()
    cells = ds.get_cell_ids()
    test_cell = cells[0]
    train_cells, test_cells = ds.leave_one_cell_out_split(test_cell)

    assert test_cell not in train_cells
    assert test_cells == [test_cell]
    assert len(train_cells) + len(test_cells) == len(cells)
