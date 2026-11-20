# NHL Prediction Project

## Project layout

- `data/raw/` — downloaded or collected source data. Treat these files as inputs.
- `data/processed/` — generated rolling-average and model-ready datasets.
- `src/` — reusable pipeline code for transforming the data.
- `notebooks/` — exploratory work, model analysis, and the original data-pulling notebooks.

## Rebuild the datasets

Use Python 3.12 and the pinned dependencies. From the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r NHL_Game_Data/requirements.txt
source .venv/bin/activate
cd NHL_Game_Data
```

For a new download:

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

## Historical date recovery

The archived raw files had blank dates because the collector read `gameDate`
from a game object instead of the containing schedule day. The collector now
persists the resolved date. Raw archives remain unchanged; the derived
`data/processed/nhl_game_dates.csv` recovers dates from the NHL schedule API for
all 8,510 archived games (2019–20 through 2025–26). To regenerate that lookup:

```bash
python src/recover_game_dates.py
```

Then run `rolling_averages.py`, `model_prep.py`, and `train_model.py` in order.
The supplied lookup lets those steps run offline. Missing dates, duplicate
team-game keys, unmatched rolling rows, and invalid home/away pairs cause
explicit failures. Rolling windows use game dates (including postponements),
exclude the current game, and reset each season. Teams with fewer than ten
previous games are excluded from training. Source statistics are unchanged.

## Model and evaluation

The target is the NHL's explicit `wins` field, including overtime and shootout.
Team-summary goal totals exclude the deciding shootout goal: comparing those
totals mislabeled 305 home wins as losses. Win flags were checked against
schedule scores for every archived game. Outcome fields are never predictors.

The pipeline compares 24 matchup differences against 48 separate home/away
rolling features, with logistic-regression C in 0.01, 0.1, 1, and 10. It never
combines differences with their redundant source columns. Median imputation
and scaling are fitted separately within each training fold. Class weights
are unbalanced to retain the observed home-win prior; the intercept captures
home advantage.

Selection minimizes mean validation log loss across expanding season folds:
train 2019–20, validate 2020–21; extend training through that season and repeat
through 2024–25. Entire seasons stay together, with strictly separated dates.
2025–26 is excluded from selection and evaluated once by this run. Its rolling
features use results of earlier games in that season, representing predictions
made as games arrive, not a forecast of the entire season before opening day.
Earlier project experiments already examined portions of 2025–26, so this is
not a pristine prospective benchmark. New future results are needed for that.

Current results: 7,336 eligible games after excluding 1,174 warmup games. The
selected model uses 24 matchup features with C=0.01. On the 1,145-game 2025–26
holdout, accuracy is 0.537, AUC 0.568, log loss 0.686, and Brier score 0.246.
The training-prior baseline has accuracy 0.521, log loss 0.693, and Brier score
0.250. These are modest improvements, not evidence of a strong forecasting
advantage. Lower log loss and Brier score are better; the reliability plot
provides a separate view of calibration. No post-hoc calibration is fitted
using holdout outcomes.

Artifacts in `models/`:

- `evaluation.json`: selection candidates, per-season metrics, baseline,
  feature list, environment versions, row counts, and SHA-256 of model input.
- `holdout_model.joblib` and `holdout_predictions.csv`: evaluation-only model
  trained on 6,191 earlier-season games and its held-out predictions.
- `logistic_regression.joblib`: production pipeline refitted on all 7,336
  eligible games, including the holdout after evaluation. Do not use this
  artifact to reproduce holdout scores.
- `feature_coefficients.csv`: standardized coefficients from the final model.
- `calibration.png`: holdout reliability plot with ten equal-count bins.

Use `pipeline.feature_names_in_` to select the ordered input columns when
predicting. Load joblib artifacts only from trusted sources and use the pinned
environment that generated them. The pipeline supports predictions only once
both teams have ten games of current-season history.

## Numerical environment and checks

The former system Python 3.9 / NumPy 2.0.2 environment emitted matrix-multiply
warnings even on finite synthetic inputs. This is consistent with the
[NumPy Accelerate issue](https://github.com/numpy/numpy/issues/28687).
The isolated Python 3.12 / NumPy 2.3.5 environment passes the full training run
with runtime and convergence warnings treated as errors; no warnings are
suppressed. Dependencies are pinned in `requirements.txt`.

From the repository root, with the environment active:

```bash
python -W error::RuntimeWarning -m unittest discover -s NHL_Game_Data/tests -v
```

Tests cover schedule-date fallback, missing dates, postponed-game ordering,
season resets, invalid inputs, shootout outcomes, holdout isolation during
selection, and final versus evaluation training counts. Probability evaluation
follows [scikit-learn's calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).
