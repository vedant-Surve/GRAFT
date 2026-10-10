
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GINEConv, MessagePassing


# --------------------------------------------------
# 1. Existing MLP model
# --------------------------------------------------
class MLP(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 16,
        out_channels: int = 2,
    ):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)


# --------------------------------------------------
# 2. Existing Edge-GNN model
# --------------------------------------------------
class EdgeGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 16,
        edge_dim: int = 4,
        out_channels: int = 2,
    ):
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


# --------------------------------------------------
# 3. New EdgeGraphSAGE convolution layer
# --------------------------------------------------
class EdgeSAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
    ):
        super().__init__(aggr="mean")

        self.lin_neigh = Linear(in_channels, out_channels)
        self.lin_edge = Linear(edge_dim, out_channels)
        self.lin_self = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )
        out = out + self.lin_self(x)
        return out

    def message(self, x_j, edge_attr):
        return self.lin_neigh(x_j) + self.lin_edge(edge_attr)


# --------------------------------------------------
# 4. New EdgeGraphSAGE account-level model
# --------------------------------------------------
class EdgeGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 64,
        edge_dim: int = 4,
        out_channels: int = 2,
    ):
        super().__init__()

        self.conv1 = EdgeSAGEConv(
            in_channels,
            hidden_channels,
            edge_dim,
        )

        self.conv2 = EdgeSAGEConv(
            hidden_channels,
            hidden_channels,
            edge_dim,
        )

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr).relu()
        return self.lin(x)


# --------------------------------------------------
# 5. New transaction-level classifier
# --------------------------------------------------

class TransactionEdgeClassifier(torch.nn.Module):
    def __init__(self, input_dim: int = 140):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.mlp(x)
