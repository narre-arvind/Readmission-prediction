"""Train baseline models, tune the strongest model, and save artifacts."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from preprocess import load_and_clean
from features import add_features, make_preprocessor

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]


def build_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Construct comparable pipelines; SMOTE is fitted only inside training folds."""
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "logistic_regression": Pipeline([("preprocess", make_preprocessor(X_train)), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE))]),
        "random_forest": Pipeline([("preprocess", make_preprocessor(X_train)), ("model", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        "xgboost": Pipeline([("preprocess", make_preprocessor(X_train)), ("model", XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=2))]),
    }


def main() -> None:
    X, y = load_and_clean(ROOT / "data/diabetic_data.csv")
    X = add_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    models = build_models(X_train, y_train)
    fitted = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
    # Tune XGBoost with stratified ROC-AUC CV; all transforms remain inside each fold.
    search = RandomizedSearchCV(fitted["xgboost"], {"model__max_depth": [3, 4, 5], "model__learning_rate": [0.03, 0.08, 0.15], "model__n_estimators": [150, 250, 350], "model__min_child_weight": [1, 3, 5]}, n_iter=5, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), random_state=RANDOM_STATE, n_jobs=1, refit=True)
    search.fit(X_train, y_train)
    fitted["xgboost_tuned"] = search.best_estimator_
    feature_schema = {column: bool(pd.api.types.is_numeric_dtype(X_train[column])) for column in X_train.columns}
    artifact = {"models": fitted, "X_test": X_test, "y_test": y_test, "feature_schema": feature_schema, "best_model": "xgboost_tuned", "best_params": search.best_params_}
    joblib.dump(artifact, ROOT / "models/evaluation_bundle.joblib", compress=3)
    deployable = {"models": {"xgboost_tuned": fitted["xgboost_tuned"]}, "X_test": X_test, "y_test": y_test, "feature_schema": feature_schema, "best_model": "xgboost_tuned", "best_params": search.best_params_}
    joblib.dump(deployable, ROOT / "models/model_bundle.joblib", compress=3)
    print(f"Rows after cleaning: {len(X)}; positive rate: {y.mean():.4f}")
    print(f"Best parameters: {search.best_params_}")


if __name__ == "__main__":
    main()
