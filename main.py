
import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="AML Fraud Detection API",
    description=(
        "AML fraud detection using independent machine-learning "
        "models for account-level and transaction-level analysis."
    ),
    version="2.0.0",
)

_prediction_service = None


def get_prediction_service():
    global _prediction_service

    if _prediction_service is None:
        from inference.prediction_service import PredictionService

        _prediction_service = PredictionService(MODELS_DIR)

    return _prediction_service


def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured in .env")

    return genai.Client(api_key=api_key)


def read_optional_alerts(alerts_file):
    """Read the optional alerts CSV."""
    if alerts_file is None:
        return pd.DataFrame()

    alerts_df = pd.read_csv(alerts_file.file)
    alerts_df.columns = alerts_df.columns.str.strip()
    return alerts_df


def read_input_files(accounts_file, transactions_file, alerts_file=None):
    """Read uploaded CSV files and normalize their column names."""
    accounts_df = pd.read_csv(accounts_file.file)
    transactions_df = pd.read_csv(transactions_file.file)
    alerts_df = read_optional_alerts(alerts_file)

    accounts_df.columns = accounts_df.columns.str.strip()
    transactions_df.columns = transactions_df.columns.str.strip()

    if accounts_df.empty:
        raise ValueError("The accounts CSV contains no records.")

    if transactions_df.empty:
        raise ValueError("The transactions CSV contains no records.")

    if "ACCOUNT_ID" not in accounts_df.columns:
        raise ValueError("The accounts CSV must contain ACCOUNT_ID.")

    return accounts_df, transactions_df, alerts_df


def build_alert_summary(alerts_df, transactions_df):
    """
    Summarize supplied alerts.

    ALERT_ID is matched only when both files contain ALERT_ID.
    This does not assume that ALERT_ID is a transaction identifier.
    Alerts are investigation indicators, not proof of fraud.
    """
    if alerts_df.empty:
        return {
            "alerts_provided": False,
            "total_alerts": 0,
            "matched_transaction_alerts": 0,
            "unmatched_transaction_alerts": 0,
            "matching_method": None,
            "alert_columns": [],
            "note": "No alert records were provided.",
        }

    total_alerts = int(len(alerts_df))

    if (
        "ALERT_ID" in alerts_df.columns
        and "ALERT_ID" in transactions_df.columns
    ):
        alert_ids = set(
            alerts_df["ALERT_ID"].dropna().astype(str).unique()
        )
        transaction_alert_ids = set(
            transactions_df["ALERT_ID"].dropna().astype(str).unique()
        )

        matched = len(alert_ids & transaction_alert_ids)
        unmatched = len(alert_ids - transaction_alert_ids)
        matching_method = "ALERT_ID"
    else:
        matched = None
        unmatched = None
        matching_method = None

    return {
        "alerts_provided": True,
        "total_alerts": total_alerts,
        "matched_transaction_alerts": matched,
        "unmatched_transaction_alerts": unmatched,
        "matching_method": matching_method,
        "alert_columns": alerts_df.columns.tolist(),
        "note": (
            "Matching is based on shared ALERT_ID values when available. "
            "If ALERT_ID is unavailable in either file, matching counts "
            "are not calculated. Alerts do not by themselves confirm fraud."
        ),
    }


def build_account_ground_truth(accounts_df, transactions_df):
    """Derive account labels from transaction IS_FRAUD labels."""
    required = {
        "SENDER_ACCOUNT_ID",
        "RECEIVER_ACCOUNT_ID",
        "IS_FRAUD",
    }
    missing = required - set(transactions_df.columns)

    if missing:
        raise ValueError(
            "Cannot calculate account accuracy. Missing transaction "
            "columns: " + ", ".join(sorted(missing))
        )

    account_ids = accounts_df["ACCOUNT_ID"].astype(str)

    tx = transactions_df[
        ["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "IS_FRAUD"]
    ].copy()

    tx["SENDER_ACCOUNT_ID"] = tx["SENDER_ACCOUNT_ID"].astype(str)
    tx["RECEIVER_ACCOUNT_ID"] = tx["RECEIVER_ACCOUNT_ID"].astype(str)
    tx["IS_FRAUD"] = (
        pd.to_numeric(tx["IS_FRAUD"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    fraudulent_accounts = set(
        tx.loc[tx["IS_FRAUD"] == 1, "SENDER_ACCOUNT_ID"]
    )
    fraudulent_accounts.update(
        tx.loc[tx["IS_FRAUD"] == 1, "RECEIVER_ACCOUNT_ID"]
    )

    return account_ids.isin(fraudulent_accounts).astype(int).to_numpy()


def calculate_model_accuracy(result, accounts_df, transactions_df):
    y_true = build_account_ground_truth(accounts_df, transactions_df)

    if len(y_true) != len(result):
        raise ValueError(
            "Ground-truth account count does not match prediction count."
        )

    model_columns = {
        "xgboost": "XGBOOST_PREDICTION",
        "mlp": "MLP_PREDICTION",
        "edge_gnn": "EDGE_GNN_PREDICTION",
        "edge_graphsage": "EDGE_GRAPHSAGE_PREDICTION",
    }

    accuracy = {}

    for model_name, prediction_col in model_columns.items():
        if prediction_col in result.columns:
            accuracy[model_name] = float(
                accuracy_score(y_true, result[prediction_col].to_numpy())
            )

    return {
        "model_accuracy": accuracy,
        "ground_truth_fraud_accounts": int(y_true.sum()),
        "ground_truth_non_fraud_accounts": int(len(y_true) - y_true.sum()),
    }


def model_stats(result, probability_col, prediction_col):
    """Calculate account-level statistics for a single model."""
    prob = pd.to_numeric(result[probability_col], errors="coerce")
    pred = pd.to_numeric(result[prediction_col], errors="coerce")

    return {
        "flagged_accounts": int((pred == 1).sum()),
        "average_fraud_probability": (
            float(prob.mean()) if prob.notna().any() else None
        ),
        "high_risk_accounts": int((prob >= 0.70).sum()),
        "medium_risk_accounts": int(
            ((prob >= 0.50) & (prob < 0.70)).sum()
        ),
        "low_risk_accounts": int((prob < 0.50).sum()),
    }


def available_model_summaries(result):
    """Return summaries only for models present in the prediction output."""
    model_columns = {
        "xgboost": (
            "XGBOOST_FRAUD_PROBABILITY",
            "XGBOOST_PREDICTION",
        ),
        "mlp": (
            "MLP_FRAUD_PROBABILITY",
            "MLP_PREDICTION",
        ),
        "edge_gnn": (
            "EDGE_GNN_FRAUD_PROBABILITY",
            "EDGE_GNN_PREDICTION",
        ),
        "edge_graphsage": (
            "EDGE_GRAPHSAGE_FRAUD_PROBABILITY",
            "EDGE_GRAPHSAGE_PREDICTION",
        ),
    }

    summaries = {}

    for name, (prob_col, pred_col) in model_columns.items():
        if prob_col in result.columns and pred_col in result.columns:
            summaries[name] = model_stats(result, prob_col, pred_col)

    return summaries


def json_safe_records(df):
    """Convert NaN/inf values to JSON nulls before returning records."""
    safe_df = df.replace([float("inf"), float("-inf")], float("nan"))
    safe_df = safe_df.astype(object).where(pd.notna(safe_df), None)
    return safe_df.to_dict(orient="records")


def get_account_prediction_result(accounts_df, transactions_df):
    return get_prediction_service().predict(accounts_df, transactions_df)


def get_transaction_prediction_result(accounts_df, transactions_df):
    service = get_prediction_service()

    if not hasattr(service, "predict_transactions"):
        raise RuntimeError(
            "PredictionService does not implement predict_transactions(). "
            "Update inference/prediction_service.py first."
        )

    result = service.predict_transactions(accounts_df, transactions_df)

    required = {
        "TRANSACTION_FRAUD_PROBABILITY",
        "TRANSACTION_PREDICTION",
    }
    missing = required - set(result.columns)

    if missing:
        raise RuntimeError(
            "predict_transactions() is missing expected output columns: "
            + ", ".join(sorted(missing))
            + ". Check inference/prediction_service.py."
        )

    return result


@app.get("/")
def root():
    return {
        "message": "AML Fraud Detection API is running",
        "status": "ok",
        "version": "2.0.0",
        "endpoints": [
            "/predict",
            "/predict-transactions",
            "/summary",
            "/llm-summary",
            "/health",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "fraud-detection-api",
    }


@app.post("/predict")
async def predict(
    accounts_file: UploadFile = File(...),
    transactions_file: UploadFile = File(...),
    alerts_file: UploadFile | None = File(default=None),
):
    try:
        accounts_df, transactions_df, alerts_df = read_input_files(
            accounts_file, transactions_file, alerts_file
        )

        result = get_account_prediction_result(
            accounts_df, transactions_df
        )
        alert_summary = build_alert_summary(alerts_df, transactions_df)

        preferred_columns = [
            "ACCOUNT_ID",
            "XGBOOST_FRAUD_PROBABILITY",
            "XGBOOST_PREDICTION",
            "MLP_FRAUD_PROBABILITY",
            "MLP_PREDICTION",
            "EDGE_GNN_FRAUD_PROBABILITY",
            "EDGE_GNN_PREDICTION",
            "EDGE_GRAPHSAGE_FRAUD_PROBABILITY",
            "EDGE_GRAPHSAGE_PREDICTION",
        ]
        columns = [c for c in preferred_columns if c in result.columns]

        return {
            "status": "success",
            "prediction_level": "account",
            "accounts_processed": int(len(result)),
            "alert_summary": alert_summary,
            "models_available": list(available_model_summaries(result).keys()),
            "results": json_safe_records(result[columns]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict-transactions")
async def predict_transactions(
    accounts_file: UploadFile = File(...),
    transactions_file: UploadFile = File(...),
    alerts_file: UploadFile | None = File(default=None),
):
    try:
        accounts_df, transactions_df, alerts_df = read_input_files(
            accounts_file,
            transactions_file,
            alerts_file,
        )

        result = get_transaction_prediction_result(
            accounts_df,
            transactions_df,
        )

        prediction = pd.to_numeric(
            result["TRANSACTION_PREDICTION"],
            errors="coerce",
        )
        probability = pd.to_numeric(
            result["TRANSACTION_FRAUD_PROBABILITY"],
            errors="coerce",
        )

        scored = prediction.notna()

        return {
            "status": "success",
            "prediction_level": "transaction",
            "total_transactions": int(len(result)),
            "scored_transactions": int(scored.sum()),
            "predicted_fraud_transactions": int(
                (scored & (prediction == 1)).sum()
            ),
            "predicted_legitimate_transactions": int(
                (scored & (prediction == 0)).sum()
            ),
            "unscored_transactions": int((~scored).sum()),
            "average_fraud_probability": (
                float(probability.mean())
                if probability.notna().any()
                else None
            ),
            "alert_summary": build_alert_summary(
                alerts_df, transactions_df
            ),
            "results": json_safe_records(result),
            "note": (
                "Alerts are investigation indicators. They are not "
                "automatically used as model features and do not prove fraud."
            ),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/summary")
async def summary(
    accounts_file: UploadFile = File(...),
    transactions_file: UploadFile = File(...),
    alerts_file: UploadFile | None = File(default=None),
):
    try:
        accounts_df, transactions_df, alerts_df = read_input_files(
            accounts_file, transactions_file, alerts_file
        )

        result = get_account_prediction_result(
            accounts_df, transactions_df
        )
        accuracy = calculate_model_accuracy(
            result, accounts_df, transactions_df
        )
        alert_summary = build_alert_summary(alerts_df, transactions_df)

        accuracy_percent = {
            name: round(value * 100, 2)
            for name, value in accuracy["model_accuracy"].items()
        }

        top_risky_accounts = {}
        top_columns = {
            "xgboost": (
                "XGBOOST_FRAUD_PROBABILITY",
                "XGBOOST_PREDICTION",
            ),
            "mlp": (
                "MLP_FRAUD_PROBABILITY",
                "MLP_PREDICTION",
            ),
            "edge_gnn": (
                "EDGE_GNN_FRAUD_PROBABILITY",
                "EDGE_GNN_PREDICTION",
            ),
            "edge_graphsage": (
                "EDGE_GRAPHSAGE_FRAUD_PROBABILITY",
                "EDGE_GRAPHSAGE_PREDICTION",
            ),
        }

        for model_name, (prob_col, pred_col) in top_columns.items():
            if prob_col in result.columns and pred_col in result.columns:
                top_risky_accounts[model_name] = json_safe_records(
                    result.nlargest(10, prob_col)[
                        ["ACCOUNT_ID", prob_col, pred_col]
                    ]
                )

        return {
            "status": "success",
            "prediction_level": "account",
            "accounts_processed": int(len(result)),
            "accuracy_note": (
                "Account-level labels are derived by marking an account "
                "fraudulent if it appears as sender or receiver in at "
                "least one supplied transaction with IS_FRAUD = 1. "
                "This derived metric is not an independent validation set."
            ),
            "alert_summary": alert_summary,
            "model_accuracy_percent": accuracy_percent,
            "ground_truth": {
                "fraud_accounts": accuracy["ground_truth_fraud_accounts"],
                "non_fraud_accounts": accuracy[
                    "ground_truth_non_fraud_accounts"
                ],
            },
            "model_summary": available_model_summaries(result),
            "top_risky_accounts": top_risky_accounts,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm-summary")
async def llm_summary(
    accounts_file: UploadFile = File(...),
    transactions_file: UploadFile = File(...),
    alerts_file: UploadFile | None = File(default=None),
):
    client = None

    try:
        accounts_df, transactions_df, alerts_df = read_input_files(
            accounts_file, transactions_file, alerts_file
        )

        result = get_account_prediction_result(
            accounts_df, transactions_df
        )
        accuracy = calculate_model_accuracy(
            result, accounts_df, transactions_df
        )
        alert_summary = build_alert_summary(alerts_df, transactions_df)

        analysis = {
            "accounts_processed": int(len(result)),
            "alert_summary": alert_summary,
            "model_accuracy_percent": {
                name: round(value * 100, 2)
                for name, value in accuracy["model_accuracy"].items()
            },
            "ground_truth": {
                "fraud_accounts": accuracy["ground_truth_fraud_accounts"],
                "non_fraud_accounts": accuracy[
                    "ground_truth_non_fraud_accounts"
                ],
            },
            "models": available_model_summaries(result),
            "top_risky_accounts": {},
        }

        top_columns = {
            "xgboost": "XGBOOST_FRAUD_PROBABILITY",
            "mlp": "MLP_FRAUD_PROBABILITY",
            "edge_gnn": "EDGE_GNN_FRAUD_PROBABILITY",
            "edge_graphsage": "EDGE_GRAPHSAGE_FRAUD_PROBABILITY",
        }

        for model_name, prob_col in top_columns.items():
            if prob_col in result.columns:
                cols = ["ACCOUNT_ID", prob_col]
                pred_col = prob_col.replace(
                    "_FRAUD_PROBABILITY", "_PREDICTION"
                )
                if pred_col in result.columns:
                    cols.append(pred_col)

                analysis["top_risky_accounts"][model_name] = (
                    json_safe_records(result.nlargest(10, prob_col)[cols])
                )

        prompt = f"""
You are an AML and financial fraud analysis assistant.

Analyze the machine-learning results and alert statistics below.

Rules:
- Analyze each available model independently.
- Do not calculate or mention an ensemble or averaged model.
- Do not invent facts, transaction patterns, motives, or relationships.
- Do not claim that an account is definitely fraudulent.
- Model probabilities are risk scores, not proof of fraud.
- Accuracy is calculated at account level using the supplied IS_FRAUD
  labels to derive account labels; this is not independent validation.
- Do not claim one model is better unless supported by the reported metrics.
- Alerts are investigation indicators, not proof of confirmed fraud.
- Do not infer alert severity or meaning from counts alone.
- Recommend proportionate human investigation and enhanced due diligence.
- Clearly distinguish model-generated risk from confirmed financial crime.
- If a model is absent from the input, do not invent results for it.

Use these sections:
1. Overall Risk Assessment
2. Key Findings
3. Model Analysis
4. Alert Analysis
5. Investigation Priorities
6. Conclusion

Machine-learning analysis:
{json.dumps(analysis, indent=2, default=str)}
"""

        client = get_gemini_client()
        try:
            interaction = client.interactions.create(
                model="gemini-3.8-flash",
                input=prompt,
            )
            llm_summary_text = interaction.output_text
        finally:
            client.close()
            client = None

        return {
            "status": "success",
            "ml_analysis": analysis,
            "llm_summary": llm_summary_text,
        }

    except Exception as e:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

        raise HTTPException(status_code=500, detail=str(e))
