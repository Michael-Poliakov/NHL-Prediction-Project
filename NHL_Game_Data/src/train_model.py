"""Train and evaluate the NHL game outcome model chronologically."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "nhl_model_data.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"


def feature_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("home_rolling_", "away_rolling_", "matchup_")
    return [c for c in df.columns if c.startswith(prefixes)]


def train_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    df = df.copy()
    df["home_team_win"] = (df["home_goals_for"] > df["away_goals_for"]).astype(int)
    features = feature_columns(df)
    df = df.dropna(subset=features).reset_index(drop=True)
    if len(df) < 30:
        raise ValueError("Not enough complete rolling-history games to train reliably")

    X, y = df[features], df["home_team_win"]
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    folds = min(5, max(2, len(X_train) // 30))
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=TimeSeriesSplit(n_splits=folds), scoring="roc_auc")
    print(f"Features: {len(features)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print(f"Baseline accuracy: {accuracy_score(y_test, baseline.predict(X_test)):.3f}")
    print(f"Time-series CV ROC AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(classification_report(y_test, predictions))
    return pipeline, pd.DataFrame({"feature": features, "coefficient": pipeline.named_steps["model"].coef_[0]})


if __name__ == "__main__":
    model, coefficients = train_and_evaluate(pd.read_csv(DATA_PATH))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    coefficients.assign(abs_coefficient=coefficients["coefficient"].abs()) \
        .sort_values("abs_coefficient", ascending=False) \
        .to_csv(MODEL_DIR / "feature_coefficients.csv", index=False)
    print(f"Saved model to {MODEL_PATH}")
