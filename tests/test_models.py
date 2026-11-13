"""Unit tests for baseline and PISTGAT battery models."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from pistgat_bench.models.ekf import ExtendedKalmanFilterSOH
from pistgat_bench.models.lstm import LSTMBatteryModel
from pistgat_bench.models.pistgat import PISTGATModel


def test_ekf_predictions():
    ekf = ExtendedKalmanFilterSOH(nominal_capacity_ah=2.0)
    features = np.random.uniform(2.8, 4.1, size=(20, 6)).astype(np.float32)
    preds = ekf.predict(features)
    assert len(preds) == 20
    assert np.all((preds >= 0.3) & (preds <= 1.1))


def test_lstm_forward_and_backward():
    model = LSTMBatteryModel(input_dim=6, hidden_dim=32, num_layers=2)
    batch_x = torch.randn(8, 10, 6)  # (batch_size=8, window=10, features=6)

    out = model(batch_x)
    assert out.shape == (8,)

    target = torch.rand(8)
    criterion = torch.nn.MSELoss()
    loss = criterion(out, target)
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_pistgat_forward_tensor():
    """Test PISTGAT running on standard 3D sequence tensors."""
    model = PISTGATModel(
        input_dim=6,
        model_dim=32,
        num_heads=2,
        transformer_layers=1,
        gat_layers=1,
        gat_heads=2
    )
    x = torch.randn(4, 8, 6)  # (nodes=4, window=8, features=6)
    out = model(x)
    assert out.shape == (4,)
    assert torch.all(torch.isfinite(out))


def test_pistgat_forward_pyg_graph():
    """Test PISTGAT running on PyTorch Geometric Data object."""
    model = PISTGATModel(
        input_dim=6,
        model_dim=32,
        num_heads=2,
        transformer_layers=1,
        gat_layers=1,
        gat_heads=2
    )
    num_nodes = 6
    window_size = 8
    x = torch.randn(num_nodes, window_size, 6)
    # Adjacent series edge connectivity: 0-1, 1-2, 2-3, 3-4, 4-5
    edge_src = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    edge_dst = [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    graph_data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    out = model(graph_data)

    assert out.shape == (num_nodes,)
    assert torch.all(torch.isfinite(out))

    # Test gradient backprop
    loss = out.sum()
    loss.backward()
    assert model.regressor[-1].weight.grad is not None
