# Titanic Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/{your-username}/titanic-classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/{your-username}/titanic-classifier/actions/workflows/tests.yml)

An end-to-end machine learning pipeline for predicting survival on the Titanic dataset using scikit-learn and pandas.

## Features

- Data loading and preprocessing pipeline
- Feature engineering (categorical encoding, missing value imputation)
- Model training with multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting)
- Model evaluation with metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- Hyperparameter tuning using grid search
- Model persistence and loading
- Command-line interface for training and prediction
- Comprehensive test suite
- GitHub Actions CI for automated testing

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/{your-username}/titanic-classifier.git
   cd titanic-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

   Or for development:
   ```bash
   pip install -e ".[dev]"
   ```

## Usage

### Training a model

```bash
titanic-classifier train --model random_forest --output models/titanic_rf.pkl
```

### Making predictions

```bash
titanic-classifier predict --input data/test.csv --model models/titanic_rf.pkl --output predictions.csv
```

### Python API

```python
from titanic_classifier.data import load_titanic_data
from titanic_classifier.model import train_model, evaluate_model
from titanic_classifier.evaluate import plot_confusion_matrix

# Load data
X_train, X_test, y_train, y_test = load_titanic_data()

# Train model
model = train_model(X_train, y_train, model_type='random_forest')

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Plot confusion matrix
plot_confusion_matrix(model, X_test, y_test)
```

## Project Structure

```
titanic-classifier/
├── data/                     # Data storage (not tracked by git)
├── titanic_classifier/       # Source code
│   ├── __init__.py
│   ├── data.py               # Data loading and preprocessing
│   ├── model.py              # Model training and persistence
│   └── evaluate.py           # Evaluation and visualization
├── tests/                    # Unit tests
├── models/                   # Trained models (not tracked by git)
├── README.md
├── LICENSE
├── .gitignore
├── CONTRIBUTING.md
├── pyproject.toml
└── requirements.txt
```

## Tech Stack

- **Language**: Python 3.8+
- **Data Processing**: pandas
- **Machine Learning**: scikit-learn
- **Model Persistence**: joblib
- **Testing**: pytest
- **CI/CD**: GitHub Actions
- **Packaging**: setuptools

## Roadmap

- [x] Basic data loading and preprocessing
- [x] Model training with Logistic Regression
- [x] Model training with Random Forest
- [x] Model evaluation metrics
- [ ] Hyperparameter tuning with Optuna
- [ ] Deep learning model (TensorFlow/PyTorch)
- [ ] Web API with FastAPI
- [ ] Docker containerization
- [ ] Interactive web dashboard with Streamlit

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Acknowledgments

- The Titanic dataset is provided by Kaggle (https://www.kaggle.com/c/titanic)
- Inspired by numerous Titanic survival prediction tutorials and kernels