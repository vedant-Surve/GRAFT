from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .models import (
    EdgeGNN,
    EdgeGraphSAGE,
    MLP,
    TransactionEdgeClassifier,
)
from .preprocessing import (
    prepare_graph,
    prepare_edge_graphsage_inputs,
    prepare_transaction_features,
)


class PredictionService:
    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self.device = torch.device("cpu")

        # Limit CPU thread usage.
        torch.set_num_threads(2)

        # Configurable transaction inference chunk size.
        self.transaction_batch_size = 1024

        # ----------------------------------------------------
        # Model 1: XGBoost
        # ----------------------------------------------------
        self.xgb = joblib.load(
            self.models_dir / "aml_fraud_xgboost_random_search_cv.pkl"
        )

        # ----------------------------------------------------
        # Model 2: MLP
        # ----------------------------------------------------
        self.mlp = MLP()
        self.mlp.load_state_dict(
            torch.load(
                self.models_dir / "best_mlp.pt",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.mlp.eval()

        # ----------------------------------------------------
        # Model 3: Edge-GNN
        # ----------------------------------------------------
        self.edge_gnn = EdgeGNN()
        self.edge_gnn.load_state_dict(
            torch.load(
                self.models_dir / "best_edge_gnn.pt",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.edge_gnn.eval()

        # Existing scaler and feature columns.
        self.scaler = joblib.load(self.models_dir / "scaler.pkl")
        self.feature_columns = joblib.load(
            self.models_dir / "feature_columns.pkl"
        )

        if getattr(self.xgb, "n_features_in_", None) != len(
            self.feature_columns
        ):
            raise ValueError(
                "XGBoost feature count does not match feature_columns.pkl"
            )

        if getattr(self.scaler, "n_features_in_", None) != len(
            self.feature_columns
        ):
            raise ValueError(
                "Scaler feature count does not match feature_columns.pkl"
            )

        # ----------------------------------------------------
        # Model 4: EdgeGraphSAGE
        # ----------------------------------------------------
        self.graphsage_checkpoint = torch.load(
            self.models_dir / "GRAFT_final_edge_graphsage.pth",
            map_location=self.device,
            weights_only=True,
        )

        self.graphsage = EdgeGraphSAGE(
            in_channels=int(
                self.graphsage_checkpoint["num_node_features"]
            ),
            hidden_channels=int(
                self.graphsage_checkpoint["hidden_channels"]
            ),
            edge_dim=int(self.graphsage_checkpoint["edge_dim"]),
            out_channels=int(
                self.graphsage_checkpoint["num_classes"]
            ),
        )

        self.graphsage.load_state_dict(
            self.graphsage_checkpoint["model_state_dict"]
        )
        self.graphsage.eval()

        self.graphsage_threshold = float(
            self.graphsage_checkpoint.get("threshold", 0.5)
        )

        # ----------------------------------------------------
        # Model 5: Transaction classifier
        # ----------------------------------------------------
        self.transaction_checkpoint = torch.load(
            self.models_dir / "GRAFT_final_transaction_classifier.pth",
            map_location=self.device,
            weights_only=True,
        )

        input_dim = int(self.transaction_checkpoint["input_dim"])

        if input_dim != 140:
            raise ValueError(
                f"Expected transaction input_dim=140, got {input_dim}"
            )

        self.transaction_model = TransactionEdgeClassifier(
            input_dim=input_dim
        )

        self.transaction_model.load_state_dict(
            self.transaction_checkpoint["model_state_dict"]
        )
        self.transaction_model.eval()

        self.transaction_threshold = float(
            self.transaction_checkpoint.get("threshold", 0.5)
        )

    # ========================================================
    # ACCOUNT-LEVEL PREDICTIONS
    # ========================================================

    def predict(
        self,
        accounts_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
    ) -> pd.DataFrame:

        prepared = prepare_graph(
            accounts_df,
            transactions_df,
            self.models_dir / "scaler.pkl",
            self.models_dir / "feature_columns.pkl",
        )

        x = prepared.x_scaled

        # Model 1: XGBoost
        xgb_prob = self.xgb.predict_proba(x)[:, 1]

        with torch.inference_mode():
            x_tensor = torch.as_tensor(
                x,
                dtype=torch.float32,
                device=self.device,
            )

            # Model 2: MLP
            mlp_logits = self.mlp(x_tensor)
            mlp_prob = torch.softmax(
                mlp_logits, dim=1
            )[:, 1].cpu().numpy()

            # Model 3: Edge-GNN
            edge_attr = torch.as_tensor(
                prepared.edge_features,
                dtype=torch.float32,
                device=self.device,
            )

            edge_logits = self.edge_gnn(
                prepared.graph.x.to(self.device),
                prepared.graph.edge_index.to(self.device),
                edge_attr,
            )

            edge_prob = torch.softmax(
                edge_logits, dim=1
            )[:, 1].cpu().numpy()

            # Model 4: EdgeGraphSAGE
            graphsage_input = prepare_edge_graphsage_inputs(
                accounts_df,
                transactions_df,
                self.graphsage_checkpoint,
            )

            graphsage_logits = self.graphsage(
                graphsage_input.graph.x.to(self.device),
                graphsage_input.graph.edge_index.to(self.device),
                graphsage_input.graph.edge_attr.to(self.device),
            )

            graphsage_prob = torch.softmax(
                graphsage_logits, dim=1
            )[:, 1].cpu().numpy()

        result = pd.DataFrame(
            {
                "ACCOUNT_ID": prepared.account_ids,
                "XGBOOST_FRAUD_PROBABILITY": xgb_prob,
                "MLP_FRAUD_PROBABILITY": mlp_prob,
                "EDGE_GNN_FRAUD_PROBABILITY": edge_prob,
                "EDGE_GRAPHSAGE_FRAUD_PROBABILITY": graphsage_prob,
            }
        )

        result["XGBOOST_PREDICTION"] = (
            result["XGBOOST_FRAUD_PROBABILITY"] >= 0.5
        ).astype(int)

        result["MLP_PREDICTION"] = (
            result["MLP_FRAUD_PROBABILITY"] >= 0.5
        ).astype(int)

        result["EDGE_GNN_PREDICTION"] = (
            result["EDGE_GNN_FRAUD_PROBABILITY"] >= 0.5
        ).astype(int)

        result["EDGE_GRAPHSAGE_PREDICTION"] = (
            result["EDGE_GRAPHSAGE_FRAUD_PROBABILITY"]
            >= self.graphsage_threshold
        ).astype(int)

        return result

    # ========================================================
    # TRANSACTION-LEVEL PREDICTIONS
    # ========================================================

    def predict_transactions(
        self,
        accounts_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        history_transactions_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Predict transaction risk using sender/receiver GraphSAGE
        embeddings and 12 scaled transaction features.

        Transaction classifier inference is processed in chunks.
        history_transactions_df must contain only transactions
        available before the transactions being scored.
        """

        required = [
            "TX_ID",
            "SENDER_ACCOUNT_ID",
            "RECEIVER_ACCOUNT_ID",
        ]

        missing = [
            column
            for column in required
            if column not in transactions_df.columns
        ]

        if missing:
            raise ValueError(
                f"transactions.csv is missing required columns: {missing}"
            )

        if transactions_df.empty:
            return pd.DataFrame(
                {
                    "TX_ID": transactions_df["TX_ID"].reset_index(
                        drop=True
                    ),
                    "TRANSACTION_FRAUD_PROBABILITY": pd.Series(
                        dtype="float64"
                    ),
                    "TRANSACTION_PREDICTION": pd.Series(
                        dtype="Int64"
                    ),
                }
            )

        print("Preparing transaction graph inputs...")

        graph_inputs = prepare_edge_graphsage_inputs(
            accounts_df,
            transactions_df,
            self.graphsage_checkpoint,
        )

        account_to_idx = {
            account_id: index
            for index, account_id in enumerate(
                graph_inputs.account_ids
            )
        }

        # Compute graph embeddings once per request.
        print("Computing GraphSAGE account embeddings...")

        with torch.inference_mode():
            x = graph_inputs.graph.x.to(self.device)
            edge_index = graph_inputs.graph.edge_index.to(self.device)
            edge_attr = graph_inputs.graph.edge_attr.to(self.device)

            embeddings = self.graphsage.conv1(
                x, edge_index, edge_attr
            ).relu()

            embeddings = self.graphsage.conv2(
                embeddings, edge_index, edge_attr
            ).relu()

        print("Preparing transaction features...")

        tx_inputs = prepare_transaction_features(
            transactions_df,
            self.transaction_checkpoint,
            history_transactions_df=history_transactions_df,
        )

        sender_idx = transactions_df[
            "SENDER_ACCOUNT_ID"
        ].map(account_to_idx)

        receiver_idx = transactions_df[
            "RECEIVER_ACCOUNT_ID"
        ].map(account_to_idx)

        valid = sender_idx.notna() & receiver_idx.notna()
        valid_positions = np.flatnonzero(valid.to_numpy())

        result = pd.DataFrame(
            {
                "TX_ID": transactions_df["TX_ID"].reset_index(
                    drop=True
                ),
            }
        )

        result["TRANSACTION_FRAUD_PROBABILITY"] = np.nan
        result["TRANSACTION_PREDICTION"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )

        total_valid = len(valid_positions)

        if total_valid == 0:
            print(
                "No transactions have both sender and receiver "
                "accounts in the graph."
            )
            return result

        # Convert indexed values once, then reuse them per chunk.
        sender_indices = sender_idx.loc[valid].to_numpy(
            dtype=np.int64
        )
        receiver_indices = receiver_idx.loc[valid].to_numpy(
            dtype=np.int64
        )

        feature_array = tx_inputs.x_scaled[valid.to_numpy()]

        batch_size = max(1, int(self.transaction_batch_size))
        all_probabilities = np.empty(
            total_valid,
            dtype=np.float32,
        )

        print(
            f"Running transaction classifier on {total_valid} "
            f"transactions in chunks of {batch_size}..."
        )

        # Process one chunk at a time instead of building one
        # large classifier input tensor for all transactions.
        with torch.inference_mode():
            for start in range(0, total_valid, batch_size):
                end = min(start + batch_size, total_valid)

                batch_sender = torch.as_tensor(
                    sender_indices[start:end],
                    dtype=torch.long,
                    device=self.device,
                )

                batch_receiver = torch.as_tensor(
                    receiver_indices[start:end],
                    dtype=torch.long,
                    device=self.device,
                )

                batch_features = torch.as_tensor(
                    feature_array[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )

                sender_embeddings = embeddings[batch_sender]
                receiver_embeddings = embeddings[batch_receiver]

                classifier_input = torch.cat(
                    [
                        sender_embeddings,
                        receiver_embeddings,
                        batch_features,
                    ],
                    dim=1,
                )

                if classifier_input.shape[1] != 140:
                    raise ValueError(
                        "Transaction classifier expected 140 features, "
                        f"got {classifier_input.shape[1]}"
                    )

                logits = self.transaction_model(classifier_input)

                batch_probabilities = torch.softmax(
                    logits, dim=1
                )[:, 1]

                all_probabilities[start:end] = (
                    batch_probabilities.cpu().numpy()
                )

                print(
                    f"Transaction prediction progress: "
                    f"{end}/{total_valid}"
                )

                # Release chunk-specific tensors before next iteration.
                del (
                    batch_sender,
                    batch_receiver,
                    batch_features,
                    sender_embeddings,
                    receiver_embeddings,
                    classifier_input,
                    logits,
                    batch_probabilities,
                )

        result.loc[
            valid_positions,
            "TRANSACTION_FRAUD_PROBABILITY",
        ] = all_probabilities

        result.loc[
            valid_positions,
            "TRANSACTION_PREDICTION",
        ] = (
            all_probabilities >= self.transaction_threshold
        ).astype(int)

        print("Transaction predictions completed.")

        return result
