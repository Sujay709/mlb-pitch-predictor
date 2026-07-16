from datetime import date, timedelta

import pandas as pd
from pybaseball import statcast_pitcher, playerid_lookup
from config import (
    PITCHER_NAME, SEASONS, START_DATE, END_DATE,
    RAW_COLUMNS, DATA_RAW_DIR,
)

CACHE_MAX_AGE_DAYS = 1  


def resolve_pitcher_id(full_name: str) -> int:
    first, last = full_name.split(" ", 1)
    lookup = playerid_lookup(last, first)

    if lookup.empty:
        raise ValueError(f"No player found for '{full_name}")
    return int(lookup.iloc[0]["key_mlbam"])


def _season_date_bounds(season: int) -> tuple[str, str]:
    """Clip a season's pull range to the overall START_DATE/END_DATE window."""
    season_start = max(f"{season}-01-01", START_DATE)
    season_end = min(f"{season}-12-31", END_DATE)
    return season_start, season_end


def _is_cache_stale(cache_path, season: int) -> bool:
    """Completed seasons are never stale. The in-progress season expires after
    CACHE_MAX_AGE_DAYS so we periodically pick up newly played games."""
    if season < date.today().year:
        return False  

    age = date.today() - date.fromtimestamp(cache_path.stat().st_mtime)
    return age > timedelta(days=CACHE_MAX_AGE_DAYS)


def pull_season(pitcher_id: int, season: int) -> pd.DataFrame:
    cache_path = DATA_RAW_DIR / f"pitcher_{pitcher_id}_{season}.parquet"
    season_start, season_end = _season_date_bounds(season)

    if cache_path.exists() and not _is_cache_stale(cache_path, season):
        print(f"Loading cached {season} data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Pulling statcast data for pitcher_id {pitcher_id}, {season} "
          f"({season_start} to {season_end})...")

    df = statcast_pitcher(season_start, season_end, pitcher_id)

    if df.empty:
        raise ValueError(f"statcast_pitcher returned no rows for {season}, check id/range")

    available_cols = [c for c in RAW_COLUMNS if c in df.columns]
    missing = set(RAW_COLUMNS) - set(available_cols)
    if missing:
        print(f"Warning: columns missing from statcast response: {missing}")

    df = df[available_cols].copy()
    df.to_parquet(cache_path, index=False)
    print(f"Cached {len(df)} rows to {cache_path}")

    return df


def pull_and_cache(pitcher_id: int, seasons: list[int]) -> pd.DataFrame:
    frames = [pull_season(pitcher_id, season) for season in seasons]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    pitcher_id = resolve_pitcher_id(PITCHER_NAME)
    print(f"{PITCHER_NAME} -> MLBAM id {pitcher_id}")

    df = pull_and_cache(pitcher_id, SEASONS)
    print(df.shape)
    print(df["pitch_type"].value_counts(normalize=True))