"""Regression checks for chronological features and honest model evaluation."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from game_dates import resolve_dates
from pull_data import get_schedule
from rolling_averages import ROLLING_STATS, build_rolling_averages
from train_model import prepare_training_data, season_splits, train_and_evaluate


class PipelineTests(unittest.TestCase):
    def test_schedule_uses_containing_date(self):
        payload = {"gameWeek": [{"date": "2024-10-04", "games": [
            {"id": 2024020001, "gameType": 2, "gameState": "OFF",
             "homeTeam": {"id": 1}, "awayTeam": {"id": 2}}]}]}
        with patch("pull_data.get_json", return_value=payload):
            schedule = get_schedule(None, "2024-10-01", "2025-06-30", 0)
        self.assertEqual(schedule.iloc[0].game_date, "2024-10-04")

    def test_missing_dates_fail(self):
        with self.assertRaisesRegex(ValueError, "Missing game dates"):
            resolve_dates(pd.DataFrame({"game_id": ["2024020001"], "game_date": [None]}),
                          Path("/nonexistent-date-lookup.csv"))

    def test_rolling_uses_dates_and_resets_each_season(self):
        rows = []
        for season in (2023, 2024):
            for n in range(12):
                rows.append({"game_id": f"{season}02{12-n:04d}",
                             "game_date": pd.Timestamp(f"{season}-10-01") + pd.Timedelta(days=n),
                             "team_id": 1, "team_name": "Example",
                             **{stat: float(n+1) for stat in ROLLING_STATS}})
        _, clean = build_rolling_averages(pd.DataFrame(rows).sample(frac=1, random_state=42))
        for season in (2023, 2024):
            group = clean[clean.game_id.str.startswith(str(season))].sort_values("game_date")
            self.assertTrue(group.goals_for_avg10.iloc[:10].isna().all())
            self.assertEqual(group.goals_for_avg10.iloc[10], 5.5)
            self.assertEqual(group.goals_for_avg5.iloc[5], 3.)
            self.assertEqual(group.goals_for_avg10.iloc[11], 6.5)

    @staticmethod
    def model_data():
        rng = np.random.default_rng(42)
        rows = []
        for season in range(2020, 2024):
            for n in range(40):
                home, away = rng.normal(3, .5, size=2)
                win = n % 2
                rows.append({"game_id": f"{season}02{n+1:04d}",
                             "game_date": pd.Timestamp(f"{season}-10-01") + pd.Timedelta(days=n),
                             "home_goals_for": 2+win, "away_goals_for": 3-win,
                             "home_wins": win, "away_wins": 1-win,
                             "home_rolling_goals_for_avg10": home,
                             "away_rolling_goals_for_avg10": away,
                             "matchup_goals_for_avg10": home-away})
        return pd.DataFrame(rows)

    def test_holdout_does_not_select_model_and_final_refits_all_games(self):
        raw = self.model_data()
        df = prepare_training_data(raw)
        folds, holdout = season_splits(df)
        self.assertEqual(holdout[0], 2023)
        for _, train, test in folds:
            self.assertLess(df.loc[train, "game_date"].max(), df.loc[test, "game_date"].min())
            self.assertNotIn(2023, df.loc[test, "season"].values)
        final, evaluation, _, report, predictions = train_and_evaluate(raw)
        changed = raw.copy()
        mask = changed.game_id.str.startswith("2023")
        changed.loc[mask, ["home_goals_for", "away_goals_for"]] = changed.loc[
            mask, ["away_goals_for", "home_goals_for"]].to_numpy()
        changed.loc[mask, "home_wins"] = 1 - changed.loc[mask, "home_wins"]
        changed.loc[mask, "away_wins"] = 1 - changed.loc[mask, "away_wins"]
        other = train_and_evaluate(changed)[3]
        self.assertEqual(report["selected"], other["selected"])
        self.assertEqual(report["candidates"], other["candidates"])
        self.assertEqual(final.named_steps["scaler"].n_samples_seen_, 160)
        self.assertEqual(evaluation.named_steps["scaler"].n_samples_seen_, 120)
        self.assertEqual(len(predictions), 40)

    def test_rejects_infinite_features_and_invalid_winners(self):
        raw = self.model_data()
        raw.loc[0, "matchup_goals_for_avg10"] = np.inf
        with self.assertRaisesRegex(ValueError, "Infinite"):
            prepare_training_data(raw)
        raw = self.model_data()
        raw.loc[0, "away_goals_for"] = raw.loc[0, "home_goals_for"]
        raw.loc[0, "home_wins"] = 1
        raw.loc[0, "away_wins"] = 0
        self.assertEqual(prepare_training_data(raw).iloc[0].home_team_win, 1)
        raw.loc[0, "away_wins"] = 1
        with self.assertRaisesRegex(ValueError, "exactly one winner"):
            prepare_training_data(raw)


if __name__ == "__main__":
    unittest.main()
