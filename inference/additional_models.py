"""Architectures and checkpoint loaders for the three additional GRAFT models.

These models are kept separate from the existing XGBoost/MLP/EdgeGNN until
end-to-end preprocessing is wired and validated against the training notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, GATConv


class EdgeSAGEConv(MessagePassing):
    """Exact edge-aware SAGE convolution used by the training notebook."""
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(aggr="mean")
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out + self.lin_self(x)

    def message(self, x_j, edge_attr):
        return self.lin_neigh(x_j) + self.lin_edge(edge_attr)


class EdgeGraphSAGE(nn.Module):
    def __init__(self, in_channels: int = 8, hidden_channels: int = 64,
                 edge_dim: int = 4, out_channels: int = 2):
        super().__init__()
        self.conv1 = EdgeSAGEConv(in_channels, hidden_channels, edge_dim)
        self.conv2 = EdgeSAGEConv(hidden_channels, hidden_channels, edge_dim)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return self.lin(x)

    def embed_nodes(self, x, edge_index, edge_attr):
        """Return the 64-dim node embeddings used by the transaction classifier."""
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x


class EdgeGAT(nn.Module):
    """Architecture reconstructed from the notebook's GAT experiment."""
    def __init__(self, in_channels: int = 8, hidden_channels: int = 16,
                 edge_dim: int = 4, out_channels: int = 2):
        super().__init__()
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=4, concat=True,
            dropout=0.5, edge_dim=edge_dim,
        )
        self.conv2 = GATConv(
            hidden_channels * 4, hidden_channels, heads=1, concat=False,
            dropout=0.5, edge_dim=edge_dim,
        )
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        return self.lin(x)


class TransactionEdgeClassifier(nn.Module):
    """MLP taking [sender embedding, receiver embedding, 12 tx features]."""
    def __init__(self, input_dim: int = 140):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.mlp(x)


def _torch_load(path: Path, device: torch.device) -> Any:
    # Checkpoints are trusted project artifacts supplied for this integration.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except (TypeError, RuntimeError, ValueError):
        return torch.load(path, map_location=device, weights_only=False)


class AdditionalModelBundle:
    """Load the three added checkpoints, preserving notebook thresholds."""
    def __init__(self, models_dir: str | Path, device: str = "cpu"):
        self.models_dir = Path(models_dir)
        self.device = torch.device(device)

        sage_ckpt = _torch_load(
            self.models_dir / "GRAFT_final_edge_graphsage.pth", self.device
        )
        self.graphsage = EdgeGraphSAGE(
            in_channels=int(sage_ckpt["num_node_features"]),
            hidden_channels=int(sage_ckpt["hidden_channels"]),
            edge_dim=int(sage_ckpt["edge_dim"]),
            out_channels=int(sage_ckpt["num_classes"]),
        ).to(self.device)
        self.graphsage.load_state_dict(sage_ckpt["model_state_dict"])
        self.graphsage.eval()
        self.graphsage_threshold = float(sage_ckpt["threshold"])
        self.graphsage_checkpoint = sage_ckpt

        gat_state = _torch_load(self.models_dir / "best_edge_gat.pt", self.device)
        self.gat = EdgeGAT().to(self.device)
        self.gat.load_state_dict(gat_state)
        self.gat.eval()
        # The notebook evaluated GAT at threshold 0.50; no tuned threshold was saved.
        self.gat_threshold = 0.50

        tx_ckpt = _torch_load(
            self.models_dir / "GRAFT_final_transaction_classifier.pth", self.device
        )
        self.transaction_classifier = TransactionEdgeClassifier(
            input_dim=int(tx_ckpt["input_dim"])
        ).to(self.device)
        self.transaction_classifier.load_state_dict(tx_ckpt["model_state_dict"])
        self.transaction_classifier.eval()
        self.transaction_threshold = float(tx_ckpt["threshold"])
        self.transaction_checkpoint = tx_ckpt

    @torch.inference_mode()
    def predict_graph_nodes(self, x, edge_index, edge_attr):
        """Inputs must already use notebook feature order/scaling/graph construction."""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32, device=self.device)

        sage_prob = torch.softmax(
            self.graphsage(x, edge_index, edge_attr), dim=1
        )[:, 1]
        gat_prob = torch.softmax(
            self.gat(x, edge_index, edge_attr), dim=1
        )[:, 1]
        return {
            "graphsage_probability": sage_prob.cpu().numpy(),
            "graphsage_prediction": (sage_prob >= self.graphsage_threshold).long().cpu().numpy(),
            "gat_probability": gat_prob.cpu().numpy(),
            "gat_prediction": (gat_prob >= self.gat_threshold).long().cpu().numpy(),
        }

    @torch.inference_mode()
    def predict_transactions(self, transaction_input):
        """Input must be 140-wide: sender embedding + receiver embedding + scaled tx features."""
        x = torch.as_tensor(
            transaction_input, dtype=torch.float32, device=self.device
        )
        if x.ndim != 2 or x.shape[1] != 140:
            raise ValueError(
                f"Transaction classifier expects shape (n, 140); got {tuple(x.shape)}"
            )
        prob = torch.softmax(self.transaction_classifier(x), dim=1)[:, 1]
        return {
            "transaction_fraud_probability": prob.cpu().numpy(),
            "transaction_prediction": (
                prob >= self.transaction_threshold
            ).long().cpu().numpy(),
        }
