"""
Model evaluation and visualization module.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.base import BaseEstimator


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate a trained model and return comprehensive metrics.

    Args:
        model: Trained sklearn estimator.
        X_test: Test features array.
        y_test: True test labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }

    return metrics


def print_evaluation_report(metrics: dict) -> None:
    """
    Print a formatted evaluation report.

    Args:
        metrics: Dictionary returned by evaluate_model().
    """
    print("=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("-" * 50)
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("-" * 50)
    print("Classification Report:")
    print(metrics["classification_report"])
    print("=" * 50)


def plot_confusion_matrix(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        model: Trained sklearn estimator.
        X_test: Test features array.
        y_test: True test labels.
        title: Title for the plot.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Died", "Survived"],
        yticklabels=["Died", "Survived"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    return fig


def plot_feature_importance(
    model: BaseEstimator,
    feature_names: list,
    top_n: int = 10,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained sklearn estimator.
        feature_names: List of feature names.
        top_n: Number of top features to display.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"Model {type(model).__name__} does not have feature_importances_"
        )

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig