import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
  PITCHER_NAME, SEASONS, TRAIN_SEASONS, VAL_SEASON,
  DATA_PROCESSED_DIR, MODELS_DIR,
)

def load_features(pitcher_id: int) -> pd.DataFrame:
  path = DATA_PROCESSED_DIR / f"pitcher_{pitcher_id}_features.parquet"
  if not path.exists():
      raise FileNotFoundError(
          f"Missing processed features: {path}. Run features.py first."
        )
  return pd.read_parquet(path)

CATEGORICAL_FEATURES = [
  "count_state", "platoon_matchup", "prev_pitch_type", "prev_pitch_type_2",
  "stand", "p_throws", "inning_topbot",
]

def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
  numeric_features = ["balls", "strikes", "outs_when_up", "inning",
    "runners_on_base", "score_diff", "prev_pitch_speed",
    "is_first_pitch_of_atbat",
  ] + [c for c in df.columns if c.startswith("game_share_")]

  feature_cols = CATEGORICAL_FEATURES + numeric_features

  preprocessor = ColumnTransformer(
    transformers=[
      ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
      ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
            ]), numeric_features),
        ]
    )
  return preprocessor, feature_cols

if __name__ == "__main__":
  from data_pull import resolve_pitcher_id

  pitcher_id = resolve_pitcher_id(PITCHER_NAME)
  df = load_features(pitcher_id)

  train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
  val_df = df[df["season"] == VAL_SEASON].copy()

  preprocessor, feature_cols = build_preprocessor(df)

  X_train, y_train = train_df[feature_cols], train_df["pitch_type"]

  model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, C=0.1, class_weight=None)),
    ])
  model.fit(X_train, y_train)

  MODELS_DIR.mkdir(parents=True, exist_ok=True)
  out_path = MODELS_DIR / f"pitcher_{pitcher_id}_logreg.joblib"
  joblib.dump(model, out_path)
  print(f"Trained model saved to {out_path}")