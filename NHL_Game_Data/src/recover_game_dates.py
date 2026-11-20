"""Recover authoritative schedule dates into a derived lookup; retain raw inputs."""
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from pull_data import RAW_DIR, get_schedule, session
from game_dates import DATE_PATH


def recover(path):
    """Fetch authoritative schedule dates for the game IDs in one archived season file."""
    original = pd.read_csv(path, dtype={"game_id": str})
    year = int(path.stem[-8:-4])
    with session() as client:
        schedule = get_schedule(client, f"{year}-07-01", f"{year+1}-06-30", .05)
    dates = original[["game_id"]].merge(
        schedule[["game_id", "game_date"]], on="game_id", how="left", validate="one_to_one")
    if dates["game_date"].isna().any():
        raise ValueError(f"Incomplete recovered schedule: {path.name}")
    dates["source"] = "https://api-web.nhle.com/v1/schedule/"
    print(f"Recovered {len(dates)} dates for {year}", flush=True)
    return dates


if __name__ == "__main__":
    files = sorted(RAW_DIR.glob("nhl_schedule_*.csv"))
    if not files:
        raise FileNotFoundError("No archived schedules found")
    with ThreadPoolExecutor(max_workers=4) as pool:
        dates = pd.concat(pool.map(recover, files), ignore_index=True)
    if dates["game_id"].duplicated().any():
        raise ValueError("Duplicate recovered game IDs")
    DATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dates.sort_values("game_id").to_csv(DATE_PATH, index=False)
