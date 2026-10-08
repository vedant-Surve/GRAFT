# AML Fraud Inference Pipeline

Standalone inference layer for the trained XGBoost, MLP, and Edge-GNN artifacts.

## Expected input

Place the original AMLSim files in `sample_data/`:

- `accounts.csv`
- `transactions.csv`
- `alerts.csv` (kept for future API/reporting use; the current three-model inference only needs accounts + transactions)

## Run

```bash
pip install -r requirements.txt
python test_inference.py
```

The script writes `predictions.csv` with account-level probabilities and predictions from all three models plus a simple mean-probability ensemble.

## Important

This inference code intentionally reproduces the preprocessing visible in the training notebook:

- 8 account/node features
- StandardScaler from training
- unique sender/receiver transaction-pair aggregation for Edge-GNN
- 4 edge features: TX_COUNT, TOTAL_AMOUNT, AVG_AMOUNT, MAX_AMOUNT
- log1p on the last 3 edge features

No retraining is performed.
