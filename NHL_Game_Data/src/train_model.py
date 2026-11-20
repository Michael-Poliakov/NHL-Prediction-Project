"""Select on expanding seasons, evaluate once on the latest, then refit for use."""
from pathlib import Path
import hashlib
import json
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data/processed/nhl_model_data.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"


def feature_columns(df, feature_set="matchup"):
    """Select either matchup-difference features or separate home and away features."""
    prefixes = ("matchup_",) if feature_set == "matchup" else ("home_rolling_", "away_rolling_")
    columns = sorted(c for c in df.columns if c.startswith(prefixes))
    if not columns:
        raise ValueError(f"No features for {feature_set}")
    return columns


def prepare_training_data(df):
    """Validate game records, derive the home-win target, and exclude games without ten-game histories."""
    df = df.copy()
    df["game_id"] = df["game_id"].astype(str)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="raise")
    if df["game_date"].isna().any() or df["game_id"].duplicated().any():
        raise ValueError("Training requires valid dates and unique game IDs")
    if not df["game_id"].str.fullmatch(r"\d{10}").all():
        raise ValueError("Expected ten-digit NHL game IDs")
    scores = df[["home_goals_for", "away_goals_for"]]
    if not np.isfinite(scores.to_numpy()).all() or (scores < 0).any().any():
        raise ValueError("Invalid final scores")
    # Team summary goals exclude the deciding shootout goal. Use the NHL's
    # explicit win flags, otherwise every tied summary is mislabeled away-win.
    wins = df[["home_wins", "away_wins"]]
    if not wins.isin([0, 1]).all().all() or not wins.sum(axis=1).eq(1).all():
        raise ValueError("Each completed game must have exactly one winner")
    decided = df["home_goals_for"] != df["away_goals_for"]
    if (df.loc[decided, "home_wins"] != (df.loc[decided, "home_goals_for"] > df.loc[decided, "away_goals_for"])).any():
        raise ValueError("Win flags disagree with non-shootout scores")
    df["home_team_win"] = df["home_wins"].astype(int)
    df["season"] = df["game_id"].str[:4].astype(int)
    features = feature_columns(df, "matchup") + feature_columns(df, "home_away")
    if np.isinf(df[features].to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values")
    # Rate-stat missing values are imputed inside each training fold. Require
    # ten actual prior games for both teams before making a prediction.
    df = df.dropna(subset=["home_rolling_goals_for_avg10", "away_rolling_goals_for_avg10"])
    return df.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)


def season_splits(df):
    """Build expanding season splits and reserve the newest season for final evaluation."""
    seasons = sorted(df["season"].unique())
    if len(seasons) < 3:
        raise ValueError("Need at least three seasons for selection and final holdout")
    splits = []
    for season in seasons[1:]:
        train = df.index[df["season"] < season]
        test = df.index[df["season"] == season]
        if df.loc[train, "game_date"].max() >= df.loc[test, "game_date"].min():
            raise ValueError("Season split overlaps dates")
        if df.loc[train, "home_team_win"].nunique() != 2 or df.loc[test, "home_team_win"].nunique() != 2:
            raise ValueError("Each season split needs both outcome classes")
        splits.append((int(season), train, test))
    return splits[:-1], splits[-1]


def make_pipeline(C):
    """Create the imputation, scaling, and logistic-regression pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        # Preserve the observed home-win prior; balanced class weights would
        # change the probability target. The intercept captures home advantage.
        ("model", LogisticRegression(C=C, max_iter=3000, random_state=42)),
    ])


def metrics(y, probabilities):
    """Calculate accuracy, ROC AUC, log loss, and Brier score for predicted outcomes."""
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite predicted probabilities")
    return {
        "accuracy": float(accuracy_score(y, probabilities >= .5)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probabilities)),
    }


def evaluate_split(df, split, features, C):
    """Fit on earlier seasons and measure performance on the following season."""
    season, train, test = split
    model = make_pipeline(C).fit(df.loc[train, features], df.loc[train, "home_team_win"])
    probability = model.predict_proba(df.loc[test, features])[:, 1]
    prior = float(df.loc[train, "home_team_win"].mean())
    report = {
        "season": season, "train_games": len(train), "test_games": len(test),
        "train_end": str(df.loc[train, "game_date"].max().date()),
        "test_start": str(df.loc[test, "game_date"].min().date()),
        "model": metrics(df.loc[test, "home_team_win"], probability),
        "baseline": metrics(df.loc[test, "home_team_win"], np.full(len(test), prior)),
    }
    return model, probability, report


def train_and_evaluate(raw):
    """Select settings on earlier seasons, evaluate the newest season, then refit on all eligible games."""
    df = prepare_training_data(raw)
    validation, holdout = season_splits(df)
    candidates = []
    for feature_set in ("matchup", "home_away"):
        features = feature_columns(df, feature_set)
        for C in (.01, .1, 1., 10.):
            folds = [evaluate_split(df, split, features, C)[2] for split in validation]
            candidates.append({"feature_set": feature_set, "C": C, "folds": folds,
                               "mean_validation_log_loss": float(np.mean([f["model"]["log_loss"] for f in folds]))})
    # Equal weight per season. Ties prefer the earlier, simpler candidate.
    best = min(candidates, key=lambda c: c["mean_validation_log_loss"])
    features = feature_columns(df, best["feature_set"])
    evaluation_model, probabilities, holdout_report = evaluate_split(df, holdout, features, best["C"])
    predictions = df.loc[holdout[2], ["game_id", "game_date", "home_team_win"]].copy()
    predictions["home_win_probability"] = probabilities
    predictions["baseline_probability"] = df.loc[holdout[1], "home_team_win"].mean()
    final_model = make_pipeline(best["C"]).fit(df[features], df["home_team_win"])
    report = {
        "target": "Home team wins, including overtime and shootout",
        "selection_metric": "Mean per-season validation log loss; lower is better",
        "selected": {k: best[k] for k in ("feature_set", "C", "mean_validation_log_loss")},
        "features": features, "candidates": candidates, "holdout": holdout_report,
        "input_games": len(raw), "eligible_games": len(df), "excluded_warmup_games": len(raw)-len(df),
        "final_training_games": len(df), "final_training_end": str(df["game_date"].max().date()),
        "final_model_includes_holdout": True,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    coefficients = pd.DataFrame({"feature": features, "coefficient": final_model.named_steps["model"].coef_[0]})
    return final_model, evaluation_model, coefficients, report, predictions


def save_calibration(predictions):
    """Save a plot comparing predicted win probabilities with observed win rates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    actual, predicted = calibration_curve(predictions.home_team_win, predictions.home_win_probability,
                                          n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(predicted, actual, "o-", label="Held-out season (10 equal-count bins)")
    ax.set(xlabel="Mean predicted home-win probability", ylabel="Observed home-win frequency",
           xlim=(0, 1), ylim=(0, 1), title="NHL model: holdout reliability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "calibration.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # Treat numerical or convergence problems as failures, never hide them.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", ConvergenceWarning)
        model, evaluation_model, coefficients, report, predictions = train_and_evaluate(pd.read_csv(DATA_PATH))
    report["input_sha256"] = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(evaluation_model, MODEL_DIR / "holdout_model.joblib")
    coefficients.assign(abs_coefficient=coefficients.coefficient.abs()).sort_values(
        "abs_coefficient", ascending=False).to_csv(MODEL_DIR / "feature_coefficients.csv", index=False)
    predictions.to_csv(MODEL_DIR / "holdout_predictions.csv", index=False)
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    save_calibration(predictions)
    print(json.dumps({"selected": report["selected"], "holdout": report["holdout"],
                      "final_training_games": report["final_training_games"]}, indent=2))
