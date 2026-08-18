"""Download NHL game, team, and player-derived team statistics.

Run from NHL_Game_Data with:
    python src/pull_data.py --season 20252026

The script is deliberately restartable: each run writes complete output files
only after the corresponding collection step finishes successfully.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


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
                if game.get("gameType") != 2 or game.get("gameState") != "OFF":
                    continue
                home, away = game.get("homeTeam", {}), game.get("awayTeam", {})
                games.append({
                    "game_id": str(game.get("id")),
                    "game_date": game.get("gameDate"),
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
    return pd.DataFrame(games).drop_duplicates("game_id").sort_values("game_id").reset_index(drop=True)


def get_team_game_stats(client: requests.Session, schedule: pd.DataFrame, delay: float) -> pd.DataFrame:
    rows = []
    for number, game in enumerate(schedule.to_dict("records"), start=1):
        url = "https://api.nhle.com/stats/rest/en/team/summary"
        data = get_json(client, f"{url}?cayenneExp=gameId={game['game_id']}", delay).get("data", [])
        for team in data:
            rows.append({
                "game_id": game["game_id"], "game_date": game["game_date"],
                "team_id": team.get("teamId"), "team_name": team.get("teamFullName"),
                "goals_for": team.get("goalsFor"), "goals_against": team.get("goalsAgainst"),
                "shots_for": team.get("shotsForPerGame"), "shots_against": team.get("shotsAgainstPerGame"),
                "faceoff_win_pct": team.get("faceoffWinPct"),
                "power_play_pct": team.get("powerPlayPct"), "penalty_kill_pct": team.get("penaltyKillPct"),
                "wins": team.get("wins"), "losses": team.get("losses"), "points": team.get("points"),
                "point_pct": team.get("pointPct"), "team_shutouts": team.get("teamShutouts"),
            })
        if number % 50 == 0:
            print(f"Team stats: {number}/{len(schedule)} games")
    return pd.DataFrame(rows).drop_duplicates(["game_id", "team_id"])


def get_player_derived_stats(client: requests.Session, schedule: pd.DataFrame, delay: float) -> pd.DataFrame:
    rows = []
    for number, game in enumerate(schedule.to_dict("records"), start=1):
        data = get_json(client, f"https://api-web.nhle.com/v1/gamecenter/{game['game_id']}/boxscore", delay)
        player_stats = data.get("playerByGameStats", {})
        for side in ("homeTeam", "awayTeam"):
            team_id = data.get(side, {}).get("id")
            players = [p for position in ("forwards", "defense", "goalies")
                       for p in player_stats.get(side, {}).get(position, [])]
            if not team_id:
                continue
            rows.append({
                "game_id": game["game_id"], "team_id": team_id,
                "hits": sum(p.get("hits", 0) or 0 for p in players),
                "blocked_shots": sum(p.get("blockedShots", 0) or 0 for p in players),
                "giveaways": sum(p.get("giveaways", 0) or 0 for p in players),
                "takeaways": sum(p.get("takeaways", 0) or 0 for p in players),
                "pims": sum(p.get("pim", 0) or 0 for p in players),
                "powerPlayGoals": sum(p.get("powerPlayGoals", 0) or 0 for p in players),
            })
        if number % 50 == 0:
            print(f"Player stats: {number}/{len(schedule)} games")
    return pd.DataFrame(rows).groupby(["game_id", "team_id"], as_index=False).sum(numeric_only=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="20252026")
    parser.add_argument("--start-date", default="2025-10-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--delay", type=float, default=0.25)
    args = parser.parse_args()

    client = session()
    schedule = get_schedule(client, args.start_date, args.end_date, args.delay)
    team = get_team_game_stats(client, schedule, args.delay)
    player = get_player_derived_stats(client, schedule, args.delay)
    complete = team.merge(player, on=["game_id", "team_id"], how="left")
    complete = complete.sort_values(["game_id", "team_id"]).reset_index(drop=True)

    season_label = f"{args.season[:4]}_{args.season[-2:]}"
    schedule.to_csv(RAW_DIR / f"nhl_schedule_{args.season}.csv", index=False)
    team.to_csv(RAW_DIR / f"nhl_team_gamelog_{season_label}.csv", index=False)
    player.to_csv(RAW_DIR / f"nhl_team_stats_from_players_{season_label}.csv", index=False)
    complete.to_csv(RAW_DIR / f"nhl_complete_team_stats_{season_label}.csv", index=False)
    print(f"Saved {len(schedule)} games and {len(complete)} complete team-game rows")


if __name__ == "__main__":
    main()
