import os
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
    description="AML fraud detection using XGBoost, MLP, and Edge-GNN",
    version="1.0.0",
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
    print("DEBUG: GEMINI_API_KEY configured =", bool(api_key))
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return genai.Client(api_key=api_key)


def build_account_ground_truth(accounts_df, transactions_df):
    required = {"SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "IS_FRAUD"}
    missing = required - set(transactions_df.columns)
    if missing:
        raise ValueError(
            "Cannot calculate accuracy. Missing transaction columns: "
            + ", ".join(sorted(missing))
        )

    account_ids = accounts_df["ACCOUNT_ID"].astype(str)
    tx = transactions_df[
        ["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "IS_FRAUD"]
    ].copy()
    tx["SENDER_ACCOUNT_ID"] = tx["SENDER_ACCOUNT_ID"].astype(str)
    tx["RECEIVER_ACCOUNT_ID"] = tx["RECEIVER_ACCOUNT_ID"].astype(str)
    tx["IS_FRAUD"] = pd.to_numeric(tx["IS_FRAUD"], errors="coerce").fillna(0).astype(int)

    fraudulent_accounts = set(tx.loc[tx["IS_FRAUD"] == 1, "SENDER_ACCOUNT_ID"])
    fraudulent_accounts.update(tx.loc[tx["IS_FRAUD"] == 1, "RECEIVER_ACCOUNT_ID"])
    return account_ids.isin(fraudulent_accounts).astype(int).to_numpy()


def calculate_model_accuracy(result, accounts_df, transactions_df):
    y_true = build_account_ground_truth(accounts_df, transactions_df)
    predictions = {
        "xgboost_accuracy": result["XGBOOST_PREDICTION"].to_numpy(),
        "mlp_accuracy": result["MLP_PREDICTION"].to_numpy(),
        "edge_gnn_accuracy": result["EDGE_GNN_PREDICTION"].to_numpy(),
    }
    if len(y_true) != len(result):
        raise ValueError("Ground-truth account count does not match prediction count.")
    return {
        "xgboost_accuracy": float(accuracy_score(y_true, predictions["xgboost_accuracy"])),
        "mlp_accuracy": float(accuracy_score(y_true, predictions["mlp_accuracy"])),
        "edge_gnn_accuracy": float(accuracy_score(y_true, predictions["edge_gnn_accuracy"])),
        "ground_truth_fraud_accounts": int(y_true.sum()),
        "ground_truth_non_fraud_accounts": int(len(y_true) - y_true.sum()),
    }


@app.get("/")
def root():
    return {"message": "AML Fraud Detection API is running", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "fraud-detection-api"}


@app.post("/predict")
async def predict(accounts_file: UploadFile = File(...), transactions_file: UploadFile = File(...)):
    try:
        accounts_df = pd.read_csv(accounts_file.file)
        transactions_df = pd.read_csv(transactions_file.file)
        result = get_prediction_service().predict(accounts_df, transactions_df)
        columns = [
            "ACCOUNT_ID",
            "XGBOOST_FRAUD_PROBABILITY", "MLP_FRAUD_PROBABILITY", "EDGE_GNN_FRAUD_PROBABILITY",
            "XGBOOST_PREDICTION", "MLP_PREDICTION", "EDGE_GNN_PREDICTION",
        ]
        return {"status": "success", "accounts_processed": len(result), "results": result[columns].to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def model_stats(result, probability_col, prediction_col):
    prob = result[probability_col]
    return {
        "flagged_accounts": int(result[prediction_col].sum()),
        "average_fraud_probability": float(prob.mean()),
        "high_risk_accounts": int((prob >= 0.70).sum()),
        "medium_risk_accounts": int(((prob >= 0.50) & (prob < 0.70)).sum()),
        "low_risk_accounts": int((prob < 0.50).sum()),
    }


@app.post("/summary")
async def summary(accounts_file: UploadFile = File(...), transactions_file: UploadFile = File(...)):
    try:
        accounts_df = pd.read_csv(accounts_file.file)
        transactions_df = pd.read_csv(transactions_file.file)
        result = get_prediction_service().predict(accounts_df, transactions_df)
        accuracy = calculate_model_accuracy(result, accounts_df, transactions_df)

        return {
            "status": "success",
            "accounts_processed": len(result),
            "accuracy_note": "Accuracy is calculated at account level. An account is labelled fraud if it appears as sender or receiver in at least one transaction with IS_FRAUD = 1.",
            "model_accuracy": {
                "xgboost_accuracy": accuracy["xgboost_accuracy"],
                "xgboost_accuracy_percent": round(accuracy["xgboost_accuracy"] * 100, 2),
                "mlp_accuracy": accuracy["mlp_accuracy"],
                "mlp_accuracy_percent": round(accuracy["mlp_accuracy"] * 100, 2),
                "edge_gnn_accuracy": accuracy["edge_gnn_accuracy"],
                "edge_gnn_accuracy_percent": round(accuracy["edge_gnn_accuracy"] * 100, 2),
            },
            "ground_truth": {
                "fraud_accounts": accuracy["ground_truth_fraud_accounts"],
                "non_fraud_accounts": accuracy["ground_truth_non_fraud_accounts"],
            },
            "model_summary": {
                "xgboost": model_stats(result, "XGBOOST_FRAUD_PROBABILITY", "XGBOOST_PREDICTION"),
                "mlp": model_stats(result, "MLP_FRAUD_PROBABILITY", "MLP_PREDICTION"),
                "edge_gnn": model_stats(result, "EDGE_GNN_FRAUD_PROBABILITY", "EDGE_GNN_PREDICTION"),
            },
            "top_risky_accounts": {
                "xgboost": result.nlargest(10, "XGBOOST_FRAUD_PROBABILITY")[["ACCOUNT_ID", "XGBOOST_FRAUD_PROBABILITY", "XGBOOST_PREDICTION"]].to_dict(orient="records"),
                "mlp": result.nlargest(10, "MLP_FRAUD_PROBABILITY")[["ACCOUNT_ID", "MLP_FRAUD_PROBABILITY", "MLP_PREDICTION"]].to_dict(orient="records"),
                "edge_gnn": result.nlargest(10, "EDGE_GNN_FRAUD_PROBABILITY")[["ACCOUNT_ID", "EDGE_GNN_FRAUD_PROBABILITY", "EDGE_GNN_PREDICTION"]].to_dict(orient="records"),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm-summary")
async def llm_summary(accounts_file: UploadFile = File(...), transactions_file: UploadFile = File(...)):
    try:
        accounts_df = pd.read_csv(accounts_file.file)
        transactions_df = pd.read_csv(transactions_file.file)
        result = get_prediction_service().predict(accounts_df, transactions_df)
        accuracy = calculate_model_accuracy(result, accounts_df, transactions_df)

        analysis = {
            "accounts_processed": len(result),
            "model_accuracy": {
                "xgboost_accuracy_percent": round(accuracy["xgboost_accuracy"] * 100, 2),
                "mlp_accuracy_percent": round(accuracy["mlp_accuracy"] * 100, 2),
                "edge_gnn_accuracy_percent": round(accuracy["edge_gnn_accuracy"] * 100, 2),
            },
            "ground_truth": {
                "fraud_accounts": accuracy["ground_truth_fraud_accounts"],
                "non_fraud_accounts": accuracy["ground_truth_non_fraud_accounts"],
            },
            "models": {
                "xgboost": model_stats(result, "XGBOOST_FRAUD_PROBABILITY", "XGBOOST_PREDICTION"),
                "mlp": model_stats(result, "MLP_FRAUD_PROBABILITY", "MLP_PREDICTION"),
                "edge_gnn": model_stats(result, "EDGE_GNN_FRAUD_PROBABILITY", "EDGE_GNN_PREDICTION"),
            },
        }

        prompt = f"""
You are an AML and financial fraud analysis assistant.

Analyze the machine-learning results below.

The system contains three independent models:
1. XGBoost
2. MLP
3. Edge-GNN

Important rules:
- Analyze each model independently.
- Do not calculate or mention an ensemble or average model.
- Do not invent facts.
- Do not claim that an account is definitely fraudulent.
- Fraud probability is a model-generated risk score, not proof of fraud.
- Accuracy is calculated at account level using the supplied transaction IS_FRAUD labels.
- Do not claim one model is better unless the reported accuracy supports it.
- Do not invent transaction patterns, motives, relationships, or AML typologies.
- Recommend human investigation and enhanced due diligence where appropriate.
- Clearly distinguish model-generated risk from confirmed financial crime.

Use these sections:
1. Overall Risk Assessment
2. Key Findings
3. Model Analysis
4. Highest-Risk Accounts
5. Investigation Priorities
6. Conclusion

Machine-learning analysis:
{analysis}
"""

        interaction = get_gemini_client().interactions.create(
            model="gemini-3.8-flash",
            input=prompt,
        )

        return {"status": "success", "ml_analysis": analysis, "llm_summary": interaction.output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
