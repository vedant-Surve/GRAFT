
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PreparedGraph:
    account_ids: pd.Series
    features: pd.DataFrame
    x_scaled: np.ndarray
    graph: Data
    edge_features: np.ndarray


@dataclass
class PreparedTransactionFeatures:
    transaction_ids: pd.Series
    features: pd.DataFrame
    x_scaled: np.ndarray


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _require_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    name: str,
) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _numeric_amounts(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["TX_AMOUNT"] = pd.to_numeric(
        result["TX_AMOUNT"],
        errors="coerce",
    ).fillna(0.0)
    return result


# ============================================================
# ORIGINAL ACCOUNT-LEVEL PREPROCESSING
# Used by XGBoost, MLP, and original Edge-GNN
# ============================================================

def build_account_features(
    accounts_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the eight account-level features."""

    _require_columns(accounts_df, ["ACCOUNT_ID"], "accounts.csv")
    _require_columns(
        transactions_df,
        [
            "SENDER_ACCOUNT_ID",
            "RECEIVER_ACCOUNT_ID",
            "TX_AMOUNT",
        ],
        "transactions.csv",
    )

    tx = _numeric_amounts(transactions_df)

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


def build_edge_features(
    transactions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the four edge features used by the original Edge-GNN."""

    _require_columns(
        transactions_df,
        [
            "SENDER_ACCOUNT_ID",
            "RECEIVER_ACCOUNT_ID",
            "TX_AMOUNT",
        ],
        "transactions.csv",
    )

    tx = _numeric_amounts(transactions_df)

    group_cols = [
        "SENDER_ACCOUNT_ID",
        "RECEIVER_ACCOUNT_ID",
    ]

    if "TX_ID" in tx.columns:
        count_column = "TX_ID"
    else:
        count_column = "TX_AMOUNT"

    edge_df = (
        tx.groupby(group_cols)
        .agg(
            TX_COUNT=(count_column, "count"),
            TOTAL_AMOUNT=("TX_AMOUNT", "sum"),
            AVG_AMOUNT=("TX_AMOUNT", "mean"),
            MAX_AMOUNT=("TX_AMOUNT", "max"),
        )
        .reset_index()
    )

    edge_features = edge_df[
        [
            "TX_COUNT",
            "TOTAL_AMOUNT",
            "AVG_AMOUNT",
            "MAX_AMOUNT",
        ]
    ].copy()

    for column in [
        "TOTAL_AMOUNT",
        "AVG_AMOUNT",
        "MAX_AMOUNT",
    ]:
        edge_features[column] = np.log1p(
            np.clip(
                edge_features[column].to_numpy(dtype=float),
                a_min=0,
                a_max=None,
            )
        )

    return edge_df, edge_features.to_numpy(dtype=np.float32)


def prepare_graph(
    accounts_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    scaler_path: str | Path,
    feature_columns_path: str | Path,
) -> PreparedGraph:
    """Prepare inputs for the original three account-level models."""

    features = build_account_features(accounts_df, transactions_df)

    feature_columns = joblib.load(feature_columns_path)
    scaler = joblib.load(scaler_path)

    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(
            "feature_columns.pkl must contain a non-empty list"
        )

    missing = [
        column
        for column in feature_columns
        if column not in features.columns
    ]

    if missing:
        raise ValueError(
            f"Cannot construct required feature columns: {missing}"
        )

    x = features[feature_columns].astype(float)
    x_scaled = scaler.transform(x)

    edge_df, edge_features = build_edge_features(transactions_df)

    account_ids = features["ACCOUNT_ID"].reset_index(drop=True)
    id_to_index = {
        account_id: index
        for index, account_id in enumerate(account_ids)
    }

    src = edge_df["SENDER_ACCOUNT_ID"].map(id_to_index)
    dst = edge_df["RECEIVER_ACCOUNT_ID"].map(id_to_index)

    valid = src.notna() & dst.notna()

    edge_df = edge_df.loc[valid].reset_index(drop=True)
    edge_features = edge_features[valid.to_numpy()]

    src = src.loc[valid].to_numpy(dtype=np.int64)
    dst = dst.loc[valid].to_numpy(dtype=np.int64)

    if len(src):
        edge_index = torch.tensor(
            np.vstack([src, dst]),
            dtype=torch.long,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    graph = Data(
        x=torch.tensor(x_scaled, dtype=torch.float32),
        edge_index=edge_index,
    )

    return PreparedGraph(
        account_ids=account_ids,
        features=x,
        x_scaled=x_scaled,
        graph=graph,
        edge_features=edge_features,
    )


# ============================================================
# NEW EDGEGRAPHSAGE PREPROCESSING
# The checkpoint contains its own scaler statistics.
# ============================================================

def prepare_edge_graphsage_inputs(
    accounts_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    checkpoint: dict,
) -> PreparedGraph:
    """
    Prepare account nodes and unique sender-receiver edges
    using the scaler and feature metadata embedded in the
    EdgeGraphSAGE checkpoint.
    """

    _require_columns(accounts_df, ["ACCOUNT_ID"], "accounts.csv")

    node_columns = checkpoint["node_features"]
    edge_columns = checkpoint["edge_features"]

    if len(node_columns) != 8:
        raise ValueError(
            f"Expected 8 node features, got {len(node_columns)}"
        )

    if len(edge_columns) != 4:
        raise ValueError(
            f"Expected 4 edge features, got {len(edge_columns)}"
        )

    features = build_account_features(accounts_df, transactions_df)
    account_ids = features["ACCOUNT_ID"].reset_index(drop=True)

    node_values = features[node_columns].to_numpy(dtype=np.float64)

    scaler_mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float64,
    )
    scaler_scale = np.asarray(
        checkpoint["scaler_scale"],
        dtype=np.float64,
    )

    if len(scaler_mean) != len(node_columns):
        raise ValueError("EdgeGraphSAGE scaler mean has an unexpected size")

    if len(scaler_scale) != len(node_columns):
        raise ValueError("EdgeGraphSAGE scaler scale has an unexpected size")

    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)
    x_scaled = ((node_values - scaler_mean) / scaler_scale).astype(
        np.float32
    )

    tx = _numeric_amounts(transactions_df)

    group_cols = [
        "SENDER_ACCOUNT_ID",
        "RECEIVER_ACCOUNT_ID",
    ]

    if "TX_ID" in tx.columns:
        count_column = "TX_ID"
    else:
        count_column = "TX_AMOUNT"

    edge_df = (
        tx.groupby(group_cols)
        .agg(
            TX_COUNT=(count_column, "count"),
            TOTAL_AMOUNT=("TX_AMOUNT", "sum"),
            AVG_AMOUNT=("TX_AMOUNT", "mean"),
            MAX_AMOUNT=("TX_AMOUNT", "max"),
        )
        .reset_index()
    )

    # EdgeGraphSAGE checkpoint uses these four edge features.
    edge_df["LOG_TOTAL_AMOUNT"] = np.log1p(
        np.clip(edge_df["TOTAL_AMOUNT"].to_numpy(dtype=float), 0, None)
    )
    edge_df["LOG_AVG_AMOUNT"] = np.log1p(
        np.clip(edge_df["AVG_AMOUNT"].to_numpy(dtype=float), 0, None)
    )
    edge_df["LOG_MAX_AMOUNT"] = np.log1p(
        np.clip(edge_df["MAX_AMOUNT"].to_numpy(dtype=float), 0, None)
    )

    account_to_idx = {
        account_id: index
        for index, account_id in enumerate(account_ids)
    }

    src = edge_df["SENDER_ACCOUNT_ID"].map(account_to_idx)
    dst = edge_df["RECEIVER_ACCOUNT_ID"].map(account_to_idx)

    valid = src.notna() & dst.notna()

    edge_df = edge_df.loc[valid].reset_index(drop=True)
    src = src.loc[valid].to_numpy(dtype=np.int64)
    dst = dst.loc[valid].to_numpy(dtype=np.int64)

    edge_attr = edge_df[edge_columns].to_numpy(dtype=np.float32)

    if len(src):
        edge_index = torch.tensor(
            np.vstack([src, dst]),
            dtype=torch.long,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    graph = Data(
        x=torch.tensor(x_scaled, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )

    return PreparedGraph(
        account_ids=account_ids,
        features=features[node_columns],
        x_scaled=x_scaled,
        graph=graph,
        edge_features=edge_attr,
    )


# ============================================================
# TRANSACTION-LEVEL PREPROCESSING
# ============================================================

TRANSACTION_BASE_FEATURES = [
    "LOG_TX_AMOUNT",
    "HOUR",
    "DAY_OF_WEEK",
    "DAY_OF_MONTH",
]

TRANSACTION_HISTORY_FEATURES = [
    "SENDER_TX_COUNT",
    "SENDER_TOTAL_AMOUNT",
    "SENDER_AVG_AMOUNT",
    "SENDER_MAX_AMOUNT",
    "RECEIVER_TX_COUNT",
    "RECEIVER_TOTAL_AMOUNT",
    "RECEIVER_AVG_AMOUNT",
    "RECEIVER_MAX_AMOUNT",
]


def build_transaction_features(
    transactions_df: pd.DataFrame,
    history_transactions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the 12 named transaction features used by the notebook.

    history_transactions_df should contain only transactions
    available before the transactions being scored. If omitted,
    the supplied transaction dataset is used as history; this is
    not guaranteed to reproduce the notebook's chronological split.
    """

    required = [
        "SENDER_ACCOUNT_ID",
        "RECEIVER_ACCOUNT_ID",
        "TX_AMOUNT",
        "TIMESTAMP",
    ]

    _require_columns(transactions_df, required, "transactions.csv")

    tx = transactions_df.copy()
    tx["TX_AMOUNT"] = pd.to_numeric(
        tx["TX_AMOUNT"],
        errors="coerce",
    ).fillna(0.0)

    timestamps = pd.to_datetime(
        tx["TIMESTAMP"],
        errors="coerce",
        utc=True,
    )

    features = pd.DataFrame(index=tx.index)

    features["LOG_TX_AMOUNT"] = np.log1p(
        np.clip(tx["TX_AMOUNT"].to_numpy(dtype=float), 0, None)
    )
    features["HOUR"] = timestamps.dt.hour.fillna(0).to_numpy()
    features["DAY_OF_WEEK"] = timestamps.dt.dayofweek.fillna(0).to_numpy()
    features["DAY_OF_MONTH"] = timestamps.dt.day.fillna(0).to_numpy()

    history = (
        tx.copy()
        if history_transactions_df is None
        else history_transactions_df.copy()
    )

    _require_columns(
        history,
        [
            "SENDER_ACCOUNT_ID",
            "RECEIVER_ACCOUNT_ID",
            "TX_AMOUNT",
        ],
        "history_transactions.csv",
    )

    history = _numeric_amounts(history)

    sender_stats = (
        history.groupby("SENDER_ACCOUNT_ID")
        .agg(
            SENDER_TX_COUNT=("TX_AMOUNT", "count"),
            SENDER_TOTAL_AMOUNT=("TX_AMOUNT", "sum"),
            SENDER_AVG_AMOUNT=("TX_AMOUNT", "mean"),
            SENDER_MAX_AMOUNT=("TX_AMOUNT", "max"),
        )
    )

    receiver_stats = (
        history.groupby("RECEIVER_ACCOUNT_ID")
        .agg(
            RECEIVER_TX_COUNT=("TX_AMOUNT", "count"),
            RECEIVER_TOTAL_AMOUNT=("TX_AMOUNT", "sum"),
            RECEIVER_AVG_AMOUNT=("TX_AMOUNT", "mean"),
            RECEIVER_MAX_AMOUNT=("TX_AMOUNT", "max"),
        )
    )

    sender_ids = tx["SENDER_ACCOUNT_ID"]
    receiver_ids = tx["RECEIVER_ACCOUNT_ID"]

    for column in sender_stats.columns:
        features[column] = sender_ids.map(sender_stats[column]).fillna(0.0)

    for column in receiver_stats.columns:
        features[column] = receiver_ids.map(receiver_stats[column]).fillna(0.0)

    # The notebook log-transforms the six amount-history fields,
    # but not the sender/receiver transaction counts.
    for column in [
        "SENDER_TOTAL_AMOUNT",
        "SENDER_AVG_AMOUNT",
        "SENDER_MAX_AMOUNT",
        "RECEIVER_TOTAL_AMOUNT",
        "RECEIVER_AVG_AMOUNT",
        "RECEIVER_MAX_AMOUNT",
    ]:
        features[column] = np.log1p(
            np.clip(
                features[column].to_numpy(dtype=float),
                0,
                None,
            )
        )

    ordered_columns = (
        TRANSACTION_BASE_FEATURES + TRANSACTION_HISTORY_FEATURES
    )

    features = features[ordered_columns]
    features = features.replace([np.inf, -np.inf], 0).fillna(0.0)

    return features


def prepare_transaction_features(
    transactions_df: pd.DataFrame,
    checkpoint: dict,
    history_transactions_df: pd.DataFrame | None = None,
) -> PreparedTransactionFeatures:
    """Scale transaction features using the saved checkpoint scaler."""

    _require_columns(
        transactions_df,
        ["TX_ID"],
        "transactions.csv",
    )

    feature_columns = checkpoint["transaction_features"]

    features = build_transaction_features(
        transactions_df,
        history_transactions_df=history_transactions_df,
    )

    missing = [
        column
        for column in feature_columns
        if column not in features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing transaction model features: {missing}"
        )

    features = features[feature_columns]

    scaler_mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float64,
    )
    scaler_scale = np.asarray(
        checkpoint["scaler_scale"],
        dtype=np.float64,
    )

    if len(scaler_mean) != len(feature_columns):
        raise ValueError("Transaction scaler mean has an unexpected size")

    if len(scaler_scale) != len(feature_columns):
        raise ValueError("Transaction scaler scale has an unexpected size")

    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)

    x_scaled = (
        (features.to_numpy(dtype=np.float64) - scaler_mean)
        / scaler_scale
    ).astype(np.float32)

    return PreparedTransactionFeatures(
        transaction_ids=transactions_df["TX_ID"].reset_index(drop=True),
        features=features.reset_index(drop=True),
        x_scaled=x_scaled,
    )
