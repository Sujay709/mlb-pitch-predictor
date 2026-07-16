from functools import cache
from matplotlib.style import available
import pandas as pd
from pybaseball import statcast_pitcher , playerid_lookup
from config import (
    PITCHER_NAME, SEASONS, START_DATE, END_DATE,
    RAW_COLUMNS, DATA_RAW_DIR,
)

def resolve_pitcher_id(full_name: str) -> int:
  first, last = full_name.split(" ", 1)
  lookup = playerid_lookup(last, first)

  if lookup.empty:
    raise ValueError(f"No player found for '{full_name}")
  return int(lookup.iloc[0]["key_mlbam"])

def pull_and_cache(pitcher_id: int, start_date: str, end_date: str) -> pd.DataFrame:
  cache_path = DATA_RAW_DIR / f"pitcher_{pitcher_id}_{start_date}_{end_date}.parquet"
  if cache_path.exists():
    print(f"Loading cached data from {cache_path}")
    return pd.read_parquet(cache_path)
  
  print(f"Pulling statcast data for pitcher_id{pitcher_id} ({start_date} to {end_date})...")

  df = statcast_pitcher(start_date, end_date, pitcher_id)

  if df.empty:
    raise ValueError("statcast_pitcher returned no rows, check id/range")

  available_cols = [c for c in RAW_COLUMNS if c in df.columns]
  missing = set(RAW_COLUMNS) - set(available_cols)
  if missing:
    print(f"Warning: columns missing from statcast response: {missing}")
  
  df = df[available_cols].copy()
  df.to_parquet(cache_path, index=False)
  print(f"Cached {len(df)} rows to {cache_path}")

  return df


if __name__ == "__main__":
  pitcher_id = resolve_pitcher_id(PITCHER_NAME)
  print(f"{PITCHER_NAME} -> MLBAM id {pitcher_id}")

  df = pull_and_cache(pitcher_id, START_DATE, END_DATE)
  print(df.shape)
  print(df["pitch_type"].value_counts(normalize=True))