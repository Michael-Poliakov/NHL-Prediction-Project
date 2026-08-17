"""Build leakage-safe rolling team features from completed NHL games."""

from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
GAMELOG = RAW_DIR / "nhl_complete_team_stats_2025_26.csv"
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
    games = games.copy()
    games["game_id"] = games["game_id"].astype(str)
    games = games.sort_values(["team_id", "game_id"], kind="stable").reset_index(drop=True)

    rolling = {}
    for window in (5, 10):
        prior = games.groupby("team_id", sort=False)[ROLLING_STATS].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        for col in ROLLING_STATS:
            rolling[f"{col}_rolling_{window}"] = prior[col]

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
    raw = pd.read_csv(GAMELOG)
    full, clean = build_rolling_averages(raw)
    full.to_csv(OUTPUT, index=False)
    clean.to_csv(CLEAN_OUTPUT, index=False)
    print(f"Wrote {len(full):,} team-game rows")
