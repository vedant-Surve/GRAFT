from pathlib import Path
import pandas as pd

from inference import PredictionService

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "sample_data"


if __name__ == "__main__":
    accounts = pd.read_csv(DATA / "accounts.csv")
    transactions = pd.read_csv(DATA / "transactions.csv")

    service = PredictionService(MODELS)
    result = service.predict(accounts, transactions)

    print("Inference completed successfully")
    print(f"Accounts scored: {len(result):,}")
    print(f"Fraud predictions (ensemble): {int(result['ENSEMBLE_PREDICTION'].sum()):,}")
    print("\nTop 10 accounts by ensemble fraud probability:")
    print(result.sort_values("ENSEMBLE_FRAUD_PROBABILITY", ascending=False).head(10).to_string(index=False))
    result.to_csv(ROOT / "predictions.csv", index=False)
    print(f"\nSaved: {ROOT / 'predictions.csv'}")
