from pathlib import Path
from datetime import date, timedelta

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
  d.mkdir(parents=True, exist_ok=True)

PITCHER_NAME = "Framber Valdez"
PITCHER_ID = None

SEASONS = [2024, 2025, 2026]
START_DATE = f"{SEASONS[0]}-03-01"

_yesterday = date.today() - timedelta(days=1)
_season_end_cap = date(SEASONS[-1], 11, 30)
END_DATE = min(_yesterday, _season_end_cap).strftime("%Y-%m-%d")

RAW_COLUMNS = [
    "pitch_type", "release_speed", "release_spin_rate",
    "balls", "strikes", "outs_when_up",
    "on_1b", "on_2b", "on_3b",
    "stand", "p_throws",
    "game_date", "at_bat_number", "pitch_number",
    "batter", "pitcher",
    "inning", "inning_topbot",
    "home_score", "away_score",
    "description", "events",
]

RARE_PITCH_THRESHOLD = 0.02
RANDOM_SEED = 42
TRAIN_SEASONS = SEASONS[:-1]
VAL_SEASON = SEASONS[-1]