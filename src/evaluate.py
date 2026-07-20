import joblib
import pandas as pd
from sklearn.metrics import (
  accuracy_score, f1_score, log_loss,
  confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import matplotlib.pyplot as plt

from config import (
  PITCHER_NAME, VAL_SEASON, DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR,
)
from train import build_preprocessor 

def load_val_data(pitcher_id: int) -> pd.DataFrame:
  path = DATA_PROCESSED_DIR / f"pitcher_{pitcher_id}_features.parquet"
  df = pd.read_parquet(path)
  return df[df["season"] == VAL_SEASON].copy()

if __name__ == "__main__":
  from data_pull import resolve_pitcher_id

  pitcher_id = resolve_pitcher_id(PITCHER_NAME)
  val_df = load_val_data(pitcher_id)

  model_path = MODELS_DIR / f"pitcher_{pitcher_id}_logreg.joblib"
  model = joblib.load(model_path)

  _, feature_cols = build_preprocessor(val_df)
  X_val, y_val = val_df[feature_cols], val_df["pitch_type"]

  preds = model.predict(X_val)
  probs = model.predict_proba(X_val)

  acc = accuracy_score(y_val, preds)
  macro_f1 = f1_score(y_val, preds, average="macro")
  ll = log_loss(y_val, probs, labels=model.classes_)

  print(f"Validation season: {VAL_SEASON}")
  print(f"accuracy={acc:.3f}  macro_f1={macro_f1:.3f}  log_loss={ll:.3f}")
  print()
  print(classification_report(y_val, preds))