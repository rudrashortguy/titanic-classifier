"""
Titanic Classifier - End-to-end ML pipeline for predicting Titanic survival.
"""

from .data import load_titanic_data, preprocess_data
from .model import train_model, predict, load_model, save_model
from .evaluate import evaluate_model, plot_confusion_matrix

__version__ = "1.0.0"
__all__ = [
    "load_titanic_data",
    "preprocess_data",
    "train_model",
    "predict",
    "load_model",
    "save_model",
    "evaluate_model",
    "plot_confusion_matrix",
]
