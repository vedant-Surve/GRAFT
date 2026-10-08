from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .models import EdgeGNN, MLP
from .preprocessing import prepare_graph


class PredictionService:
    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self.device = torch.device("cpu")

        self.xgb = joblib.load(self.models_dir / "aml_fraud_xgboost_random_search_cv.pkl")
        self.mlp = MLP()
        self.mlp.load_state_dict(torch.load(self.models_dir / "best_mlp.pt", map_location=self.device, weights_only=True))
        self.mlp.eval()

        self.edge_gnn = EdgeGNN()
        self.edge_gnn.load_state_dict(torch.load(self.models_dir / "best_edge_gnn.pt", map_location=self.device, weights_only=True))
        self.edge_gnn.eval()

        self.scaler = joblib.load(self.models_dir / "scaler.pkl")
        self.feature_columns = joblib.load(self.models_dir / "feature_columns.pkl")

        if getattr(self.xgb, "n_features_in_", None) != len(self.feature_columns):
            raise ValueError("XGBoost feature count does not match feature_columns.pkl")
        if getattr(self.scaler, "n_features_in_", None) != len(self.feature_columns):
            raise ValueError("Scaler feature count does not match feature_columns.pkl")

    def predict(self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
        prepared = prepare_graph(
            accounts_df,
            transactions_df,
            self.models_dir / "scaler.pkl",
            self.models_dir / "feature_columns.pkl",
        )

        x = prepared.x_scaled
        xgb_prob = self.xgb.predict_proba(x)[:, 1]

        with torch.inference_mode():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            mlp_logits = self.mlp(x_tensor)
            mlp_prob = torch.softmax(mlp_logits, dim=1)[:, 1].cpu().numpy()

            edge_attr = torch.tensor(prepared.edge_features, dtype=torch.float32, device=self.device)
            edge_logits = self.edge_gnn(
                prepared.graph.x.to(self.device),
                prepared.graph.edge_index.to(self.device),
                edge_attr,
            )
            edge_prob = torch.softmax(edge_logits, dim=1)[:, 1].cpu().numpy()

        result = pd.DataFrame({
            "ACCOUNT_ID": prepared.account_ids,
            "XGBOOST_FRAUD_PROBABILITY": xgb_prob,
            "MLP_FRAUD_PROBABILITY": mlp_prob,
            "EDGE_GNN_FRAUD_PROBABILITY": edge_prob,
        })
        result["XGBOOST_PREDICTION"] = (result["XGBOOST_FRAUD_PROBABILITY"] >= 0.5).astype(int)
        result["MLP_PREDICTION"] = (result["MLP_FRAUD_PROBABILITY"] >= 0.5).astype(int)
        result["EDGE_GNN_PREDICTION"] = (result["EDGE_GNN_FRAUD_PROBABILITY"] >= 0.5).astype(int)
        result["ENSEMBLE_FRAUD_PROBABILITY"] = result[
            ["XGBOOST_FRAUD_PROBABILITY", "MLP_FRAUD_PROBABILITY", "EDGE_GNN_FRAUD_PROBABILITY"]
        ].mean(axis=1)
        result["ENSEMBLE_PREDICTION"] = (result["ENSEMBLE_FRAUD_PROBABILITY"] >= 0.5).astype(int)
        return result
