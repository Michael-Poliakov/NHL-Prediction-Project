"""Build leakage-safe rolling team features from completed NHL games."""

from pathlib import Path
import pandas as pd
from game_dates import resolve_dates


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT = PROCESSED_DIR / "nhl_team_gamelog_with_rolling_averages.csv"
CLEAN_OUTPUT = PROCESSED_DIR / "nhl_team_rolling_averages_clean.csv"

ROLLING_STATS = [
    "goals_for", "goals_against", "shots_for", "shots_against",
    "faceoff_win_pct", "power_play_pct", "penalty_kill_pct", "hits",
    "blocked_shots", "giveaways", "takeaways", "pims",
]


def build_rolling_averages(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full log with raw + rolling columns, rolling-only export).

    Each row's rolling values use only games before that row. This prevents
    the result from leaking the current game's outcome into model features.
    """
    games = resolve_dates(games)
    if games.duplicated(["game_id", "team_id"]).any():
        raise ValueError("Duplicate team-game rows")
    if games.duplicated(["team_id", "game_date"]).any():
        raise ValueError("Multiple games for a team on the same date")
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="raise")
    # NHL game IDs start with the season's first calendar year. Keeping this
    # key in the rolling group prevents prior-season form carrying over the
    # offseason into a new season's 5- and 10-game windows.
    games["_season_id"] = games["game_id"].str[:4]
    games = games.sort_values(
        ["team_id", "_season_id", "game_date", "game_id"], kind="stable"
    ).reset_index(drop=True)

    rolling = {}
    for window in (5, 10):
        prior = games.groupby(["team_id", "_season_id"], sort=False)[ROLLING_STATS].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )
        for col in ROLLING_STATS:
            rolling[f"{col}_rolling_{window}"] = prior[col]

    games = games.drop(columns="_season_id")
    full = pd.concat([games, pd.DataFrame(rolling, index=games.index)], axis=1)
    full = full.sort_values(["game_id", "team_id"], kind="stable").reset_index(drop=True)

    clean = full[["game_id", "game_date", "team_id", "team_name"] + list(rolling)].copy()
    clean.columns = [
        *clean.columns[:4],
        *[c.replace("_rolling_5", "_avg5").replace("_rolling_10", "_avg10")
           for c in clean.columns[4:]],
    ]
    return full, clean


if __name__ == "__main__":
    files = sorted(RAW_DIR.glob("nhl_complete_team_stats_*.csv"))
    if not files:
        raise FileNotFoundError(f"No complete team-game files found in {RAW_DIR}")
    raw = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    full, clean = build_rolling_averages(raw)
    full.to_csv(OUTPUT, index=False)
    clean.to_csv(CLEAN_OUTPUT, index=False)
    print(f"Wrote {len(full):,} team-game rows")
