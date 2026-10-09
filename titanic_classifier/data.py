"""
Data loading and preprocessing module for Titanic dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_titanic_data(
    filepath: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Load the Titanic dataset and split into training and testing sets.

    Args:
        filepath: Path to the CSV file containing Titanic data.
            If None, uses bundled sample data.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    if filepath is None:
        filepath = _get_sample_data_path()

    df = pd.read_csv(filepath)

    # Basic validation
    if "Survived" not in df.columns:
        raise ValueError("Dataset must contain 'Survived' column")

    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, list(X.columns)


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_features: list = None,
    categorical_features: list = None,
) -> tuple:
    """
    Preprocess training and testing data with consistent transformations.

    Handles missing values and encodes categorical variables.

    Args:
        X_train: Training features DataFrame.
        X_test: Testing features DataFrame.
        numeric_features: List of numeric feature names.
        categorical_features: List of categorical feature names.

    Returns:
        Tuple of (X_train_processed, X_test_processed, transformer)
    """
    if numeric_features is None:
        numeric_features = ["Age", "Fare", "Parch", "SibSp", "Pclass"]

    if categorical_features is None:
        categorical_features = ["Sex", "Embarked", "Ticket", "Cabin"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", None),  # Will be set after fitting
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, preprocessor


def _get_sample_data_path() -> str:
    """Return path to bundled sample Titanic data."""
    import os
    sample_data = """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W (Emily Hoskins)",female,27,0,2,347742,11.1333,,S
10,1,2,"Nasser, Mrs. Nicholas (Adele Achem)",female,14,1,0,237736,30.0708,,C
"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(sample_data)
        return f.name