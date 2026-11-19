"""Create one home/away modeling row per NHL game."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ROLLING = PROCESSED_DIR / "nhl_team_rolling_averages_clean.csv"
OUTPUT = PROCESSED_DIR / "nhl_model_data.csv"

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


def prepare_model_data(
    games: pd.DataFrame, rolling: pd.DataFrame, schedules: pd.DataFrame | None = None
) -> pd.DataFrame:
    games = games.copy()
    rolling = rolling.copy()
    for df in (games, rolling):
        df["game_id"] = df["game_id"].astype(str)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="raise")

    if schedules is not None and not schedules.empty:
        schedules = schedules.copy()
        schedules["game_id"] = schedules["game_id"].astype(str)
        home = schedules[["game_id", "home_team_id"]].rename(columns={"home_team_id": "team_id"})
        home["side"] = "home"
        away = schedules[["game_id", "away_team_id"]].rename(columns={"away_team_id": "team_id"})
        away["side"] = "away"
        sides = pd.concat([home, away], ignore_index=True)
        sides["team_id"] = pd.to_numeric(sides["team_id"], errors="coerce")
        games["team_id"] = pd.to_numeric(games["team_id"], errors="coerce")
        games = games.drop(columns="side", errors="ignore").merge(
            sides, on=["game_id", "team_id"], how="left", validate="many_to_one"
        )
    elif "side" not in games.columns:
        raise ValueError("Home/away side is missing; provide schedule data with team IDs")

    if games["side"].isna().any():
        missing = games.loc[games["side"].isna(), "game_id"].nunique()
        raise ValueError(f"Could not match team rows to home/away schedule IDs for {missing} games")

    games = games.sort_values(["game_date", "game_id", "side"], kind="stable")
    games = games.merge(rolling, on=["game_id", "team_id", "team_name", "game_date"], how="left")

    rows = []
    for game_id, group in games.groupby("game_id", sort=False):
        if len(group) != 2 or set(group["side"]) != {"home", "away"}:
            continue
        out = {"game_id": game_id, "game_date": group["game_date"].iloc[0]}
        for side in ("home", "away"):
            row = group.loc[group["side"] == side].iloc[0]
            for col in BASE_STATS:
                out[f"{side}_{col}"] = row[col]
            for window in ("avg5", "avg10"):
                for stat in ROLLING_STATS:
                    out[f"{side}_rolling_{stat}_{window}"] = row[f"{stat}_{window}"]
        rows.append(out)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)

    # Matchup features make the comparison between teams explicit and reduce
    # the burden on the model to discover useful home-versus-away contrasts.
    for stat in ROLLING_STATS:
        for window in ("avg5", "avg10"):
            home = f"home_rolling_{stat}_{window}"
            away = f"away_rolling_{stat}_{window}"
            result[f"matchup_{stat}_{window}"] = result[home] - result[away]
    return result


if __name__ == "__main__":
    game_files = sorted(RAW_DIR.glob("nhl_complete_team_stats_*.csv"))
    schedule_files = sorted(RAW_DIR.glob("nhl_schedule_*.csv"))
    if not game_files:
        raise FileNotFoundError(f"No complete team-game files found in {RAW_DIR}")
    if not schedule_files:
        raise FileNotFoundError(f"No schedule files found in {RAW_DIR}")
    games = pd.concat((pd.read_csv(path) for path in game_files), ignore_index=True)
    games = games.drop_duplicates(["game_id", "team_id"], keep="last")
    schedules = pd.concat((pd.read_csv(path) for path in schedule_files), ignore_index=True)
    schedules = schedules.drop_duplicates("game_id", keep="last")
    rolling = pd.read_csv(ROLLING)
    result = prepare_model_data(games, rolling, schedules)
    result.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(result):,} game rows and {len(result.columns):,} columns")
