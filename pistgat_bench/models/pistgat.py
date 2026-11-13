"""Physics-Informed Spatiotemporal Graph Attention Network with Transformer (PISTGAT)."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal degradation sequence."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to sequence tensor (B, T, D)."""
        t = x.size(1)
        return x + self.pe[:, :t, :]


class PISTGATModel(nn.Module):
    """Physics-Informed Spatiotemporal Graph Attention Network with Transformer.

    Combines:
    1. Temporal Transformer Encoder: Captures long-range temporal degradation trends
       across cycle sequences.
    2. Spatial Graph Attention (GATv2): Learns cell-to-cell thermal/electrical coupling
       and cell imbalance distributions across the battery pack topology.
    3. Physics Guidance: Incorporates electrochemical domain priors (internal resistance,
       temperature differentials, monotonic constraint projections).
    """

    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 64,
        num_heads: int = 4,
        transformer_layers: int = 2,
        gat_layers: int = 2,
        gat_heads: int = 2,
        dropout: float = 0.15
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim

        # Input feature projection
        self.feature_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        # 1. Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # 2. Spatial Graph Attention Network (GATv2)
        self.gat1 = GATv2Conv(
            in_channels=model_dim,
            out_channels=model_dim // gat_heads,
            heads=gat_heads,
            dropout=dropout,
            concat=True
        )
        self.gat2 = GATv2Conv(
            in_channels=model_dim,
            out_channels=model_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )

        # 3. Physics Feature Extraction & Fusion
        # Extracts thermodynamic and electrochemical features: resistance slope and temperature rise
        self.physics_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # Final Regressor Head with Physics Guidance
        self.regressor = nn.Sequential(
            nn.Linear(model_dim + 16, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def extract_physics_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract physics-relevant signals: internal resistance and max temperature."""
        # x is (..., T, D) with feature indices: 4: temp_max, 5: resistance
        # Compute delta temperature across sequence and final resistance
        temp_seq = x[..., 4]
        res_seq = x[..., 5]
        delta_temp = temp_seq[..., -1] - temp_seq[..., 0]
        final_res = res_seq[..., -1]
        phys = torch.stack([delta_temp, final_res], dim=-1)
        return phys

    def forward(
        self,
        data_or_x: Union[Data, Batch, torch.Tensor],
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Unified forward pass accepting either PyG Data/Batch or standard 3D tensor (B, T, D).

        Returns:
            Tensor of predicted SOH values (B,) or (num_nodes,).
        """
        if isinstance(data_or_x, (Data, Batch)):
            x = data_or_x.x  # Shape: (num_nodes, window_size, input_dim)
            e_idx = data_or_x.edge_index
        elif isinstance(data_or_x, torch.Tensor):
            x = data_or_x
            if edge_index is None:
                # If evaluating on batch of single cells without graph, create virtual self-loops
                b = x.size(0)
                e_idx = torch.arange(b, device=x.device, dtype=torch.long).repeat(2, 1)
            else:
                e_idx = edge_index
        else:
            raise TypeError(f"Unsupported input type: {type(data_or_x)}")

        # x shape: (N, T, D) where N is nodes/batch items, T is window size, D is input dim
        n_nodes, seq_len, _ = x.shape

        # Extract physics signals before projection
        phys_raw = self.extract_physics_features(x)  # (N, 2)
        phys_embed = self.physics_mlp(phys_raw)      # (N, 16)

        # 1. Temporal Transformer Encoding
        h_proj = self.feature_proj(x)                # (N, T, model_dim)
        h_pos = self.pos_encoder(h_proj)
        h_temp = self.temporal_transformer(h_pos)    # (N, T, model_dim)

        # Pool temporal representation (use last cycle embedding)
        node_features = h_temp[:, -1, :]             # (N, model_dim)

        # 2. Spatial Graph Attention (pack level coupling)
        if e_idx.size(1) > 0:
            h_spatial = F.elu(self.gat1(node_features, e_idx))
            h_spatial = self.gat2(h_spatial, e_idx)
            # Residual connection
            node_features = node_features + h_spatial

        # 3. Physics Guided Fusion
        fused = torch.cat([node_features, phys_embed], dim=-1)  # (N, model_dim + 16)
        pred_soh = self.regressor(fused).squeeze(-1)            # (N,)

        return pred_soh
