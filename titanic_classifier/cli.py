"""
Command-line interface for Titanic Classifier.
"""

import argparse
import sys
from pathlib import Path

from .data import load_titanic_data, preprocess_data
from .model import train_model, predict, load_model, save_model
from .evaluate import evaluate_model, print_evaluation_report, plot_confusion_matrix


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Titanic Classifier - ML pipeline for survival prediction"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model",
        choices=list(train_model.__wrapped__.keywords["model_type"].choices)
        if hasattr(train_model, "__wrapped__")
        else ["logistic_regression", "random_forest", "gradient_boosting", "knn", "svm"],
        default="random_forest",
        help="Model type to train",
    )
    train_parser.add_argument(
        "--input",
        default=None,
        help="Path to CSV file with Titanic data",
    )
    train_parser.add_argument(
        "--output",
        default="models/titanic_model.pkl",
        help="Path to save the trained model",
    )
    train_parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Use grid search for hyperparameter tuning",
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model file",
    )
    predict_parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV file with features",
    )
    predict_parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Path to save predictions",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model file",
    )
    eval_parser.add_argument(
        "--input",
        default=None,
        help="Path to CSV file with labeled data",
    )
    eval_parser.add_argument(
        "--plot-confusion",
        action="store_true",
        help="Generate confusion matrix plot",
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = _parse_args()

    if args.command == "train":
        print("Loading data...")
        X_train, X_test, y_train, y_test, feature_names = load_titanic_data(
            filepath=args.input
        )

        print("Preprocessing data...")
        X_train_proc, X_test_proc, preprocessor = preprocess_data(X_train, X_test)

        print(f"Training {args.model} model...")
        model = train_model(
            X_train_proc,
            y_train,
            model_type=args.model,
            use_grid_search=args.grid_search,
        )

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_model(model, args.output)
        print(f"Model saved to {args.output}")

        print("\nEvaluating on test set...")
        metrics = evaluate_model(model, X_test_proc, y_test)
        print_evaluation_report(metrics)

        if args.plot_confusion:
            plot_confusion_matrix(model, X_test_proc, y_test)

    elif args.command == "predict":
        model = load_model(args.model)

        df = pd.read_csv(args.input)
        # Assuming first column is Survived label for evaluation
        if "Survived" in df.columns:
            y_true = df["Survived"]
            X = df.drop(columns=["Survived"])
        else:
            X = df
            y_true = None

        X_train, X_test, _, _, _ = load_titanic_data()
        X_proc, _, _ = preprocess_data(X_train, X)
        predictions = predict(model, X_proc)

        result_df = df.copy()
        result_df["Prediction"] = predictions
        if y_true is not None:
            result_df["Actual"] = y_true

        result_df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")

    elif args.command == "evaluate":
        model = load_model(args.model)
        X_train, X_test, y_train, y_test, _ = load_titanic_data(filepath=args.input)
        X_proc, X_test_proc, _ = preprocess_data(X_train, X_test)

        metrics = evaluate_model(model, X_proc, y_train)
        print_evaluation_report(metrics)

        if args.plot_confusion:
            plot_confusion_matrix(model, X_proc, y_train)

    else:
        _parse_args().print_help()


if __name__ == "__main__":
    main()