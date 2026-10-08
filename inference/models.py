from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GINEConv


class MLP(torch.nn.Module):
    def __init__(self, in_channels: int = 8, hidden_channels: int = 16, out_channels: int = 2):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)


class EdgeGNN(torch.nn.Module):
    def __init__(self, in_channels: int = 8, hidden_channels: int = 16, edge_dim: int = 4, out_channels: int = 2):
        super().__init__()
        self.conv1 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            ),
            edge_dim=edge_dim,
        )
        self.conv2 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            ),
            edge_dim=edge_dim,
        )
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        return self.lin(x)
