from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


@dataclass
class PreparedGraph:
    account_ids: pd.Series
    features: pd.DataFrame
    x_scaled: np.ndarray
    graph: Data
    edge_features: np.ndarray


def _require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def build_account_features(accounts_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the 8 node features used during training."""
    _require_columns(accounts_df, ["ACCOUNT_ID"], "accounts.csv")
    _require_columns(
        transactions_df,
        ["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT"],
        "transactions.csv",
    )

    tx = transactions_df.copy()
    tx["TX_AMOUNT"] = pd.to_numeric(tx["TX_AMOUNT"], errors="coerce").fillna(0.0)

    outgoing = (
        tx.groupby("SENDER_ACCOUNT_ID")["TX_AMOUNT"]
        .agg(
            OUT_TX_COUNT="count",
            TOTAL_OUT_AMOUNT="sum",
            AVG_OUT_AMOUNT="mean",
            MAX_OUT_AMOUNT="max",
        )
        .reset_index()
        .rename(columns={"SENDER_ACCOUNT_ID": "ACCOUNT_ID"})
    )

    incoming = (
        tx.groupby("RECEIVER_ACCOUNT_ID")["TX_AMOUNT"]
        .agg(
            IN_TX_COUNT="count",
            TOTAL_IN_AMOUNT="sum",
            AVG_IN_AMOUNT="mean",
            MAX_IN_AMOUNT="max",
        )
        .reset_index()
        .rename(columns={"RECEIVER_ACCOUNT_ID": "ACCOUNT_ID"})
    )

    features = accounts_df[["ACCOUNT_ID"]].copy()
    features = features.merge(outgoing, on="ACCOUNT_ID", how="left")
    features = features.merge(incoming, on="ACCOUNT_ID", how="left")

    node_columns = [
        "OUT_TX_COUNT",
        "TOTAL_OUT_AMOUNT",
        "AVG_OUT_AMOUNT",
        "MAX_OUT_AMOUNT",
        "IN_TX_COUNT",
        "TOTAL_IN_AMOUNT",
        "AVG_IN_AMOUNT",
        "MAX_IN_AMOUNT",
    ]
    features[node_columns] = features[node_columns].fillna(0.0)
    return features


def build_edge_features(transactions_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Reproduce the 4 Edge-GNN edge features from training."""
    _require_columns(
        transactions_df,
        ["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT"],
        "transactions.csv",
    )
    tx = transactions_df.copy()
    tx["TX_AMOUNT"] = pd.to_numeric(tx["TX_AMOUNT"], errors="coerce").fillna(0.0)

    group_cols = ["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID"]
    edge_df = (
        tx.groupby(group_cols)
        .agg(
            TX_COUNT=("TX_AMOUNT", "count"),
            TOTAL_AMOUNT=("TX_AMOUNT", "sum"),
            AVG_AMOUNT=("TX_AMOUNT", "mean"),
            MAX_AMOUNT=("TX_AMOUNT", "max"),
        )
        .reset_index()
    )

    edge_features = edge_df[["TX_COUNT", "TOTAL_AMOUNT", "AVG_AMOUNT", "MAX_AMOUNT"]].copy()
    for col in ["TOTAL_AMOUNT", "AVG_AMOUNT", "MAX_AMOUNT"]:
        edge_features[col] = np.log1p(np.clip(edge_features[col].to_numpy(dtype=float), a_min=0, a_max=None))

    return edge_df, edge_features.to_numpy(dtype=np.float32)


def prepare_graph(
    accounts_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    scaler_path: str | Path,
    feature_columns_path: str | Path,
) -> PreparedGraph:
    features = build_account_features(accounts_df, transactions_df)
    feature_columns = joblib.load(feature_columns_path)
    scaler = joblib.load(scaler_path)

    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("feature_columns.pkl must contain a non-empty list")
    if list(features.columns.drop("ACCOUNT_ID")) != feature_columns:
        # Select and reorder exactly as training did. This also makes accidental input ordering harmless.
        missing = [c for c in feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Cannot construct required feature columns: {missing}")

    x = features[feature_columns].astype(float)
    x_scaled = scaler.transform(x)

    edge_df, edge_features = build_edge_features(transactions_df)
    account_ids = list(features["ACCOUNT_ID"])
    id_to_index = {account_id: i for i, account_id in enumerate(account_ids)}

    src = edge_df["SENDER_ACCOUNT_ID"].map(id_to_index)
    dst = edge_df["RECEIVER_ACCOUNT_ID"].map(id_to_index)
    valid = src.notna() & dst.notna()
    if not valid.all():
        edge_df = edge_df.loc[valid].reset_index(drop=True)
        edge_features = edge_features[valid.to_numpy()]
        src = src.loc[valid]
        dst = dst.loc[valid]

    edge_index = torch.tensor(
        np.vstack([src.to_numpy(dtype=np.int64), dst.to_numpy(dtype=np.int64)]),
        dtype=torch.long,
    )
    graph = Data(
        x=torch.tensor(x_scaled, dtype=torch.float32),
        edge_index=edge_index,
    )

    return PreparedGraph(
        account_ids=features["ACCOUNT_ID"],
        features=x,
        x_scaled=x_scaled,
        graph=graph,
        edge_features=edge_features,
    )
