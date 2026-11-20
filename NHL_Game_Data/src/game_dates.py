"""Resolve legacy missing dates without modifying archived raw tables."""
from pathlib import Path
import pandas as pd

DATE_PATH = Path(__file__).resolve().parents[1] / "data/processed/nhl_game_dates.csv"


def resolve_dates(frame, date_path=DATE_PATH):
    frame = frame.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    dates = pd.to_datetime(frame["game_date"], errors="raise")
    if dates.isna().any() and date_path.exists():
        lookup = pd.read_csv(date_path, dtype={"game_id": str})
        if lookup["game_id"].duplicated().any():
            raise ValueError("Duplicate game IDs in recovered dates")
        lookup = lookup.set_index("game_id")["game_date"]
        dates = dates.fillna(pd.to_datetime(frame["game_id"].map(lookup), errors="raise"))
    if dates.isna().any():
        raise ValueError("Missing game dates; run src/recover_game_dates.py first")
    frame["game_date"] = dates
    return frame
