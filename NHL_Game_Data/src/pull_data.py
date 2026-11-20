"""Download NHL game, team, and player-derived team statistics.

Run from NHL_Game_Data with:
    python src/pull_data.py
    python src/pull_data.py --seasons 20192020 20202021 20212022 20222023 20232024 20242025 20252026

The script is deliberately restartable: each run writes complete output files
only after the corresponding collection step finishes successfully.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from threading import local
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
_thread_state = local()


def session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    client = requests.Session()
    client.headers.update({"User-Agent": "NHL-Prediction-Project/1.0"})
    client.mount("https://", adapter)
    return client


def worker_session() -> requests.Session:
    """Return a configured HTTP session owned by the current worker thread."""
    if not hasattr(_thread_state, "client"):
        _thread_state.client = session()
    return _thread_state.client


def get_json(client: requests.Session, url: str, delay: float) -> dict:
    response = client.get(url, timeout=30)
    response.raise_for_status()
    time.sleep(delay)
    return response.json()


def get_schedule(client: requests.Session, start_date: str, end_date: str, delay: float) -> pd.DataFrame:
    games, seen_dates = [], set()
    current = start_date
    while current and current <= end_date:
        if current in seen_dates:
            raise RuntimeError(f"Schedule pagination repeated date {current}")
        seen_dates.add(current)
        data = get_json(client, f"https://api-web.nhle.com/v1/schedule/{current}", delay)
        for week in data.get("gameWeek", []):
            for game in week.get("games", []):
                # Schedule game objects commonly omit gameDate; the containing
                # gameWeek date is the NHL's official schedule date.
                game_date = game.get("gameDate") or week.get("date")
                if (
                    not game_date or game_date < start_date or game_date > end_date
                    or game.get("gameType") != 2 or game.get("gameState") != "OFF"
                ):
                    continue
                home, away = game.get("homeTeam", {}), game.get("awayTeam", {})
                games.append({
                    "game_id": str(game.get("id")),
                    "game_date": game_date,
                    "home_team_id": home.get("id"),
                    "away_team_id": away.get("id"),
                    "home_team": home.get("abbrev"),
                    "away_team": away.get("abbrev"),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                })
        next_date = data.get("nextStartDate")
        if not next_date or next_date == current:
            break
        current = next_date
    columns = [
        "game_id", "game_date", "home_team_id", "away_team_id", "home_team",
        "away_team", "home_score", "away_score",
    ]
    return (
        pd.DataFrame(games, columns=columns)
        .drop_duplicates("game_id")
        .sort_values("game_id")
        .reset_index(drop=True)
    )


def get_team_game_stats(
    schedule: pd.DataFrame, delay: float, workers: int = 4
) -> pd.DataFrame:
    def fetch_game(game: dict) -> list[dict]:
        url = "https://api.nhle.com/stats/rest/en/team/summary"
        data = get_json(
            worker_session(), f"{url}?cayenneExp=gameId={game['game_id']}", delay
        ).get("data", [])
        game_rows = []
        for team in data:
            team_id = team.get("teamId")
            side = (
                "home" if team_id == game.get("home_team_id")
                else "away" if team_id == game.get("away_team_id")
                else None
            )
            game_rows.append({
                "game_id": game["game_id"], "game_date": game["game_date"],
                "team_id": team_id, "team_name": team.get("teamFullName"), "side": side,
                "goals_for": team.get("goalsFor"), "goals_against": team.get("goalsAgainst"),
                "shots_for": team.get("shotsForPerGame"), "shots_against": team.get("shotsAgainstPerGame"),
                "faceoff_win_pct": team.get("faceoffWinPct"),
                "power_play_pct": team.get("powerPlayPct"), "penalty_kill_pct": team.get("penaltyKillPct"),
                "wins": team.get("wins"), "losses": team.get("losses"), "points": team.get("points"),
                "point_pct": team.get("pointPct"), "team_shutouts": team.get("teamShutouts"),
            })
        return game_rows

    rows = []
    records = schedule.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, game_rows in enumerate(pool.map(fetch_game, records), start=1):
            rows.extend(game_rows)
            if number % 50 == 0:
                print(f"Team stats: {number}/{len(schedule)} games")
    return pd.DataFrame(rows).drop_duplicates(["game_id", "team_id"])


def get_player_derived_stats(
    schedule: pd.DataFrame, delay: float, workers: int = 4
) -> pd.DataFrame:
    def fetch_game(game: dict) -> list[dict]:
        data = get_json(
            worker_session(),
            f"https://api-web.nhle.com/v1/gamecenter/{game['game_id']}/boxscore",
            delay,
        )
        player_stats = data.get("playerByGameStats", {})
        game_rows = []
        for side in ("homeTeam", "awayTeam"):
            team_id = data.get(side, {}).get("id")
            players = [p for position in ("forwards", "defense", "goalies")
                       for p in player_stats.get(side, {}).get(position, [])]
            if not team_id:
                continue
            game_rows.append({
                "game_id": game["game_id"], "team_id": team_id,
                "hits": sum(p.get("hits", 0) or 0 for p in players),
                "blocked_shots": sum(p.get("blockedShots", 0) or 0 for p in players),
                "giveaways": sum(p.get("giveaways", 0) or 0 for p in players),
                "takeaways": sum(p.get("takeaways", 0) or 0 for p in players),
                "pims": sum(p.get("pim", 0) or 0 for p in players),
                "powerPlayGoals": sum(p.get("powerPlayGoals", 0) or 0 for p in players),
            })
        return game_rows

    rows = []
    records = schedule.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, game_rows in enumerate(pool.map(fetch_game, records), start=1):
            rows.extend(game_rows)
            if number % 50 == 0:
                print(f"Player stats: {number}/{len(schedule)} games")
    return pd.DataFrame(rows).groupby(["game_id", "team_id"], as_index=False).sum(numeric_only=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons", nargs="+", default=["20242025", "20252026"],
        help="Season IDs in YYYYyyyy format; defaults to the two most recent seasons.",
    )
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    client = session()
    for season in args.seasons:
        if len(season) != 8 or not season.isdigit():
            parser.error(f"Invalid season {season!r}; use YYYYyyyy, for example 20242025")
        start_year = int(season[:4])
        if int(season[4:]) != start_year + 1:
            parser.error(f"Invalid season {season!r}; ending year must follow starting year")

        # A broad date range accommodates season schedule variations; gameType=2
        # in get_schedule filters out preseason and playoff games.
        start_date = f"{start_year}-10-01"
        end_date = f"{start_year + 1}-06-30"
        schedule = get_schedule(client, start_date, end_date, args.delay)
        team = get_team_game_stats(schedule, args.delay, args.workers)
        player = get_player_derived_stats(schedule, args.delay, args.workers)
        complete = team.merge(player, on=["game_id", "team_id"], how="left")
        complete = complete.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)

        season_label = f"{season[:4]}_{season[-2:]}"
        schedule.to_csv(RAW_DIR / f"nhl_schedule_{season}.csv", index=False)
        team.to_csv(RAW_DIR / f"nhl_team_gamelog_{season_label}.csv", index=False)
        player.to_csv(RAW_DIR / f"nhl_team_stats_from_players_{season_label}.csv", index=False)
        complete.to_csv(RAW_DIR / f"nhl_complete_team_stats_{season_label}.csv", index=False)
        print(f"{season_label}: saved {len(schedule)} games and {len(complete)} team-game rows")


if __name__ == "__main__":
    main()
