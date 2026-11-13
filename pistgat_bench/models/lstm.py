"""LSTM baseline neural network for battery SOH forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBatteryModel(nn.Module):
    """Bidirectional LSTM recurrent baseline for battery State of Health regression.

    Takes a temporal window of cycle-level operational features (voltage, current,
    temperature, resistance) and predicts the final SOH for the cycle.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.input_norm = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        fc_in_dim = hidden_dim * num_directions
        self.head = nn.Sequential(
            nn.Linear(fc_in_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Tensor of shape (batch_size,) representing predicted SOH.
        """
        # x: (B, T, D)
        b, t, d = x.shape
        # Normalize along feature dimension
        x_reshaped = x.view(-1, d)
        x_norm = self.input_norm(x_reshaped).view(b, t, d)

        out, (h_n, c_n) = self.lstm(x_norm)

        # Use the representation from the last time step
        last_step = out[:, -1, :]  # (B, hidden_dim * num_directions)
        pred = self.head(last_step).squeeze(-1)  # (B,)
        return pred
