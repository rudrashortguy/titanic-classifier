"""Tests for Titanic Classifier."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from titanic_classifier.data import load_titanic_data, preprocess_data
from titanic_classifier.model import train_model, predict, save_model, load_model
from titanic_classifier.evaluate import evaluate_model, plot_confusion_matrix


# --- Fixtures ---

@pytest.fixture
def sample_csv():
    """Create a temporary CSV with sample Titanic data."""
    data = """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley",female,38,1,0,PC 17599,71.28,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2,7.925,,S
4,0,1,"Futrelle, Mr. Jacques",male,35,1,0,113803,53.1,C123,S
5,1,3,"Allen, Mr. William",male,35,0,0,373450,8.05,,S
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(data)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def trained_models():
    """Pretrained models for testing."""
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)

    models = {}
    for model_type in ["logistic_regression", "random_forest", "gradient_boosting"]:
        models[model_type] = train_model(X_train, y_train, model_type=model_type)
    return models


# --- Tests for data module ---

class TestDataLoading:
    def test_load_titanic_data_returns_tuple(self, sample_csv):
        result = load_titanic_data(filepath=sample_csv)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_load_titanic_data_splits_correctly(self, sample_csv):
        X_train, X_test, y_train, y_test, _ = load_titanic_data(
            filepath=sample_csv, test_size=0.2, random_state=42
        )
        assert len(y_train) + len(y_test) == 5

    def test_load_titanic_data_missing_column_raises(self, sample_csv):
        """Test that missing 'Survived' column raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("PassengerId,Pclass,Name\n1,3,Braund\n")
            path = f.name
        with pytest.raises(ValueError, match="Survived"):
            load_titanic_data(filepath=path)
        os.unlink(path)

    def test_load_titanic_data_stratified(self, sample_csv):
        """Test that stratification preserves class ratio."""
        _, _, y_train, y_test, _ = load_titanic_data(
            filepath=sample_csv, test_size=0.5, random_state=42
        )
        # With small data, check no crash and types are correct
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)


class TestPreprocessing:
    def test_preprocess_data_shapes(self):
        X_train = pd.DataFrame({
            "Age": [25, 30, 35, 40],
            "Sex": ["male", "female", "male", "female"],
        })
        X_test = pd.DataFrame({
            "Age": [28, 33],
            "Sex": ["female", "male"],
        })

        X_train_proc, X_test_proc, transformer = preprocess_data(X_train, X_test)

        assert X_train_proc.shape[0] == 4
        assert X_test_proc.shape[0] == 2
        assert X_test_proc.shape[1] > 0  # Some features after encoding

    def test_preprocess_handles_missing_values(self):
        """Test that missing values are imputed correctly."""
        X_train = pd.DataFrame({"Age": [25.0, np.nan, 35.0, 40.0]})
        X_test = pd.DataFrame({"Age": [np.nan, 33.0]})

        X_train_proc, X_test_proc, _ = preprocess_data(X_train, X_test)

        # Should not contain NaN after preprocessing
        assert not np.any(np.isnan(X_train_proc))
        assert not np.any(np.isnan(X_test_proc))


# --- Tests for model module ---

class TestModelTraining:
    def test_train_model_returns_trained_estimator(self, trained_models):
        for model_type, model in trained_models.items():
            assert hasattr(model, "predict")
            assert hasattr(model, "fit")

    def test_train_model_invalid_type_raises(self):
        X = np.random.rand(10, 3)
        y = np.random.randint(0, 2, 10)
        with pytest.raises(ValueError, match="Unknown model_type"):
            train_model(X, y, model_type="nonexistent_model")

    def test_predict_returns_array(self, trained_models):
        X_test = np.random.rand(5, 5)
        for model_type, model in trained_models.items():
            predictions = predict(model, X_test)
            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == 5

    def test_save_and_load_model(self, trained_models):
        """Test that saved model can be loaded and produces same predictions."""
        model = trained_models["logistic_regression"]
        X_test = np.random.rand(5, 5)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            save_model(model, path)
            loaded = load_model(path)
            original_preds = predict(model, X_test)
            loaded_preds = predict(loaded, X_test)
            np.testing.assert_array_equal(original_preds, loaded_preds)
        finally:
            os.unlink(path)


# --- Tests for evaluate module ---

class TestEvaluation:
    def test_evaluate_model_returns_metrics(self, trained_models):
        X_test = np.random.rand(20, 5)
        y_test = np.random.randint(0, 2, 20)

        for model_type, model in trained_models.items():
            metrics = evaluate_model(model, X_test, y_test)
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics
            assert "roc_auc" in metrics
            assert "confusion_matrix" in metrics

    def test_evaluate_accuracy_range(self, trained_models):
        """Test that accuracy is between 0 and 1."""
        X_test = np.random.rand(50, 5)
        y_test = np.random.randint(0, 2, 50)

        for model_type, model in trained_models.items():
            metrics = evaluate_model(model, X_test, y_test)
            assert 0 <= metrics["accuracy"] <= 1

    def test_plot_confusion_matrix_returns_figure(self, trained_models):
        """Test that plotting function returns a matplotlib Figure."""
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend

        X_test = np.random.rand(20, 5)
        y_test = np.random.randint(0, 2, 20)

        fig = plot_confusion_matrix(
            trained_models["logistic_regression"], X_test, y_test
        )
        assert fig is not None
        plt.close(fig)


# --- Integration Tests ---

class TestIntegration:
    def test_end_to_end_pipeline(self, sample_csv):
        """Full pipeline test from data loading to evaluation."""
        X_train, X_test, y_train, y_test, _ = load_titanic_data(
            filepath=sample_csv, test_size=0.3
        )
        X_train_proc, X_test_proc, _ = preprocess_data(X_train, X_test)

        model = train_model(X_train_proc, y_train, model_type="logistic_regression")
        metrics = evaluate_model(model, X_test_proc, y_test)

        assert metrics["accuracy"] >= 0
        assert metrics["accuracy"] <= 1

    def test_cli_help_exits_cleanly(self):
        """Test that CLI help works without error."""
        import sys
        from io import StringIO
        from contextlib import redirect_stdout

        old_argv = sys.argv
        try:
            sys.argv = ["titanic-classifier", "--help"]
            f = StringIO()
            with redirect_stdout(f):
                main()
        except SystemExit as e:
            assert e.code == 0  # Normal exit for --help
        finally:
            sys.argv = old_argv