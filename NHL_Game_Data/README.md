# NHL Prediction Project

## Project layout

- `data/raw/` — downloaded or collected source data. Treat these files as inputs.
- `data/processed/` — generated rolling-average and model-ready datasets.
- `src/` — reusable pipeline code for transforming the data.
- `notebooks/` — exploratory work, model analysis, and the original data-pulling notebooks.

## Rebuild the datasets

From the `NHL_Game_Data` directory:

```bash
python src/rolling_averages.py
python src/model_prep.py
python src/train_model.py
```

The scripts use paths relative to the project directory, so they can be run
from any working directory. `Model.ipynb` consumes the generated
`data/processed/nhl_model_data.csv` file.

## Data flow

```text
data/raw/nhl_complete_team_stats_2025_26.csv
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
holdout, and time-series cross-validation.
