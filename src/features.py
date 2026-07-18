import pandas as pd

from config import (
  PITCHER_NAME, SEASONS, RARE_PITCH_THRESHOLD, DATA_RAW_DIR, DATA_PROCESSED_DIR
)

def load_seasons(pitcher_id: int, seasons: list[int]) -> pd.DataFrame:
  frames = []
  for season in seasons():
    path = DATA_RAW_DIR / f"pitcher_{pitcher_id}_{season}.parquet"
    if not path.exists():
      raise FileExistsError(f"Missing cached data for {season}: {path}")
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

def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  at_bat_group = ["game_pk", "at_bat_number"]
  df["prev_pitch_type"] = df.groupby(at_bat_group)["pitch_type"].shift(1)
  df["prev_pitch_speed"] = df.groupby(at_bat_group)["release_speed"].shift(1)
  df["prev_pitch_type_2"] = df.groupby(at_bat_group)["pitch_type_2"].shift(2)
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