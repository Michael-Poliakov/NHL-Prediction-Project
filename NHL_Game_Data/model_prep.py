"""Create one home/away modeling row per NHL game."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
GAMELOG = DATA_DIR / "nhl_complete_team_stats_2025_26.csv"
ROLLING = DATA_DIR / "nhl_team_rolling_averages_clean.csv"
OUTPUT = DATA_DIR / "nhl_model_data.csv"

BASE_STATS = [
    "team_id", "team_name", "goals_for", "goals_against", "shots_for",
    "shots_against", "faceoff_win_pct", "power_play_pct", "penalty_kill_pct",
    "wins", "losses", "points", "point_pct", "team_shutouts",
]
ROLLING_STATS = [
    "goals_for", "goals_against", "shots_for", "shots_against",
    "faceoff_win_pct", "power_play_pct", "penalty_kill_pct", "hits",
    "blocked_shots", "giveaways", "takeaways", "pims",
]


def prepare_model_data(games: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    rolling = rolling.copy()
    for df in (games, rolling):
        df["game_id"] = df["game_id"].astype(str)

    # Schedule-derived home/away is not present in the team log. The two rows
    # per game are retained in source order by the collector (home, then away).
    games["team_row_number"] = games.groupby("game_id", sort=False).cumcount()
    games = games[games["team_row_number"] < 2].copy()
    games = games.merge(rolling, on=["game_id", "team_id", "team_name", "game_date"], how="left")
    games["side"] = games.groupby("game_id", sort=False).cumcount().map({0: "home", 1: "away"})

    rows = []
    for game_id, group in games.groupby("game_id", sort=False):
        if len(group) != 2:
            continue
        group = group.sort_values("team_row_number")
        out = {"game_id": game_id}
        for side, (_, row) in zip(("home", "away"), group.iterrows()):
            for col in BASE_STATS:
                out[f"{side}_{col}"] = row[col]
            for window in ("avg5", "avg10"):
                for stat in ROLLING_STATS:
                    out[f"{side}_rolling_{stat}_{window}"] = row[f"{stat}_{window}"]
            out[f"{side}_rolling_team_row_number"] = row["team_row_number"]
        rows.append(out)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    games = pd.read_csv(GAMELOG)
    rolling = pd.read_csv(ROLLING)
    result = prepare_model_data(games, rolling)
    result.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(result):,} game rows and {len(result.columns):,} columns")
