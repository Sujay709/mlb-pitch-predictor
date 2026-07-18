import pandas as pd

from config import (
  PITCHER_NAME, SEASONS, RARE_PITCH_THRESHOLD, DATA_RAW_DIR, DATA_PROCESSED_DIR
)

def load_seasons(pitcher_id: int, seasons: list[int]) -> pd.DataFrame:
  frames = []
  for season in seasons:
    path = DATA_RAW_DIR / f"pitcher_{pitcher_id}_{season}.parquet"
    if not path.exists():
      raise FileNotFoundError(f"Missing cached data for {season}: {path}")
    df = pd.read_parquet(path)
    df["season"] = season
    frames.append(df)
  
  return pd.concat(frames, ignore_index=False)

def sort_pitches(df: pd.DataFrame) -> pd.DataFrame:
  """Chronological pitch order within and across games. game_pk breaks ties
  that at_bat_number alone can't (at_bat_number resets every game)."""
  return df.sort_values(
    ["season", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

def drop_unclassified_pitches(df: pd.DataFrame) -> pd.DataFrame:
  """Statcast occasionally leaves pitch_type as NaN (unclassified pitches).
    These aren't a real pitch type we can predict or use as sequence
    history, so we drop them BEFORE building sequence features -- otherwise
    a dropped pitch leaves a false 'first pitch of at-bat' signal on the
    pitch that actually followed it."""
  before = len(df)
  df = df[df["pitch_type"].notna()].reset_index(drop=True)
  dropped = before - len(df)
  if dropped:
    print(f"Dropped {dropped} pitches with unclassified pitch_type")
  return df


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  at_bat_group = ["game_pk", "at_bat_number"]
  df["prev_pitch_type"] = df.groupby(at_bat_group)["pitch_type"].shift(1)
  df["prev_pitch_speed"] = df.groupby(at_bat_group)["release_speed"].shift(1)
  df["prev_pitch_type_2"] = df.groupby(at_bat_group)["pitch_type"].shift(2)
  df["is_first_pitch_of_atbat"] = df["prev_pitch_type"].isna()

  pitch_dummies = pd.get_dummies(df["pitch_type"], prefix="seen")
  game_key = df["game_pk"]
  running_counts = pitch_dummies.groupby(game_key).cumsum().groupby(game_key).shift(1)
  running_counts = running_counts.fillna(0)

  pitch_num_so_far = df.groupby(game_key).cumcount()

  running_share = running_counts.div(pitch_num_so_far.replace(0, pd.NA), axis=0)
  running_share = running_share.fillna(0).add_prefix("game_share_")

  df = pd.concat([df, running_share], axis=1)

  return df

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["count_state"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)

  df["runner_on_1b"] = df["on_1b"].notna()
  df["runner_on_2b"] = df["on_2b"].notna()
  df["runner_on_3b"] = df["on_3b"].notna()

  df["runners_on_base"] = (
    df["runner_on_1b"].astype(int)
    + df["runner_on_2b"].astype(int)
    + df["runner_on_3b"].astype(int)
  )

  df["platoon_matchup"] = (df["stand"] == df["p_throws"]).map(
    {True: "same", False: "opposite"}
  )
  pitching_team_score = df["home_score"].where(
    df["inning_topbot"] == "Top", df["away_score"]
  )
  batting_team_score = df["away_score"].where(
    df["inning_topbot"] == "Top", df["home_score"]
  )
  df["score_diff"] = pitching_team_score - batting_team_score
  return df

def filter_rare_pitch_types(df: pd.DataFrame, threshold = float) -> pd.DataFrame:
  freq = df["pitch_type"].value_counts(normalize=True)
  keep = freq[freq >= threshold].index

  dropped = freq[freq < threshold]
  if not dropped.empty:
    print(f"Dropping rare pitch types (< {threshold:.1%} usage):"
          f"{dict(dropped.round(4))}")
  return df[df["pitch_type"].isin(keep)].reset_index(drop=True)

def build_features(pitcher_id: int, seasons: list[int]) -> pd.DataFrame:
  df = load_seasons(pitcher_id, seasons)
  df = sort_pitches(df)
  df = drop_unclassified_pitches(df)
  df = add_sequence_features(df)
  df = add_context_features(df)
  df = filter_rare_pitch_types(df, RARE_PITCH_THRESHOLD)
  return df

if __name__ == "__main__":
  from data_pull import resolve_pitcher_id

  pitcher_id = resolve_pitcher_id(PITCHER_NAME)
  df = build_features(pitcher_id, SEASONS)

  out_path = DATA_PROCESSED_DIR / f"pitcher_{pitcher_id}_features.parquet"
  df.to_parquet(out_path, index=False)
  print(f"Saved {len(df)} rows x {df.shape[1]} cols to {out_path}")
  print(df[["season", "game_pk", "at_bat_number", "pitch_number",
            "pitch_type", "prev_pitch_type", "count_state",
            "runners_on_base", "score_diff"]].head(10))