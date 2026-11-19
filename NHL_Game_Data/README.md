# NHL Prediction Project

## Project layout

- `data/raw/` — downloaded or collected source data. Treat these files as inputs.
- `data/processed/` — generated rolling-average and model-ready datasets.
- `src/` — reusable pipeline code for transforming the data.
- `notebooks/` — exploratory work, model analysis, and the original data-pulling notebooks.

## Rebuild the datasets

From the `NHL_Game_Data` directory:

```bash
python src/pull_data.py
python src/rolling_averages.py
python src/model_prep.py
python src/train_model.py
```

The collector now downloads the 2024–25 and 2025–26 regular seasons by
default. To backfill more seasons, pass season IDs as the start year followed
by the next year, for
example:

```bash
python src/pull_data.py --seasons 20192020 20202021 20212022 20222023 20232024 20242025 20252026
```

The collector uses four workers by default and keeps a 0.25-second delay per
request. Use `--workers` to adjust concurrency if the NHL API rate-limits the
backfill.

Each season is saved as its own raw CSV set. The rolling-average and model
preparation steps automatically combine all matching complete team-game files
in `data/raw/`, so rerun those steps after collecting additional seasons.
Rolling 5- and 10-game features reset at each season boundary, so they describe
current-season form only. The first five or ten games for a team have missing
rolling values until enough games have been played.

The scripts use paths relative to the project directory, so they can be run
from any working directory. `Model.ipynb` consumes the generated
`data/processed/nhl_model_data.csv` file.

## Data flow

```text
data/raw/nhl_complete_team_stats_YYYY_YY.csv (one file per season)
        │
        ▼
src/rolling_averages.py
        │
        ├── data/processed/nhl_team_gamelog_with_rolling_averages.csv
        └── data/processed/nhl_team_rolling_averages_clean.csv
                                      │
                                      ▼
                              src/model_prep.py
                                      │
                                      ▼
                       data/processed/nhl_model_data.csv
```

Training artifacts are saved under `models/`. The model uses only complete
5- and 10-game prior histories, matchup-difference features, a chronological
holdout, and time-series cross-validation. Schedule team IDs determine home
and away rows, so CSV row order does not affect matchup construction. Rate
statistics that are undefined for a game are median-imputed inside the
training pipeline; games without a full ten-game history are excluded.
