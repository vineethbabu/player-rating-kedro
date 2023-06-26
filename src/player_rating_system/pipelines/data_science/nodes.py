import logging
from typing import Dict, Tuple

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """

    X = data[parameters["features"]]
    y = data["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Trains the Decision Tree Classification model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for performance class of player.

    Returns:
        Trained model.
    """
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    return dt


def evaluate_model(
    classifier: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series
):
    """

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has an accuracy %.3f on test data.", score)