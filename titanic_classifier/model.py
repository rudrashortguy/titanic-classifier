"""
Model training, prediction, and persistence module.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin


# Map of available model types to their classes
MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "svm": SVC(kernel="rbf", probability=True, random_state=42),
}


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    hyperparameters: dict = None,
    use_grid_search: bool = False,
) -> BaseEstimator:
    """
    Train a classifier on the preprocessed Titanic data.

    Args:
        X_train: Training features array.
        y_train: Training labels array.
        model_type: Name of the model to train.
        hyperparameters: Optional dict of hyperparameters to override defaults.
        use_grid_search: If True, perform grid search for best hyperparameters.

    Returns:
        Trained sklearn estimator.

    Raises:
        ValueError: If model_type is not supported.
    """
    if model_type not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {available}")

    base_model = MODEL_REGISTRY[model_type].clone()

    if hyperparameters:
        base_model.set_params(**hyperparameters)

    if use_grid_search and model_type == "random_forest":
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring="accuracy", n_jobs=-1
        )
        base_model = grid_search.fit(X_train, y_train)
        base_model = base_model.best_estimator_

    base_model.fit(X_train, y_train)
    return base_model


def predict(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """
    Make predictions using a trained model.

    Args:
        model: Trained sklearn estimator.
        X: Features array for prediction.

    Returns:
        Array of predicted labels.
    """
    return model.predict(X)


def predict_proba(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """
    Predict class probabilities.

    Args:
        model: Trained sklearn estimator.
        X: Features array.

    Returns:
        Array of shape (n_samples, n_classes) with probabilities.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    raise AttributeError(
        f"Model of type {type(model).__name__} does not support predict_proba"
    )


def save_model(model: BaseEstimator, filepath: str) -> None:
    """
    Save a trained model to disk.

    Args:
        model: Trained sklearn estimator.
        filepath: Path where the model will be saved.
    """
    joblib.dump(model, filepath)


def load_model(filepath: str) -> BaseEstimator:
    """
    Load a model from disk.

    Args:
        filepath: Path to the saved model.

    Returns:
        Loaded sklearn estimator.
    """
    return joblib.load(filepath)