import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit

def split_data(data: pd.DataFrame) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    # Split the data into train, test, and validation sets
    train_size = 0.7
    test_size = 0.15
    validation_size = 0.15


    X = data.drop(['reservations'], axis=1)
    Y = data["reservations"]
    # Initialize StratifiedShuffleSplit
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=validation_size + test_size, random_state=42)

    # Split the data into training and remaining data
    for train_index, remaining_index in stratified_split.split(X, Y):
        X_train, X_remaining = X.iloc[train_index], X.iloc[remaining_index]
        y_train, y_remaining = Y.iloc[train_index], Y.iloc[remaining_index]

    # Initialize another StratifiedShuffleSplit for splitting the remaining data into validation and test sets
    stratified_split_remaining = StratifiedShuffleSplit(n_splits=1, test_size=test_size / (validation_size + test_size), random_state=42)
    # Split the remaining data into validation and test sets
    for validation_index, test_index in stratified_split_remaining.split(X_remaining, y_remaining):
        X_valid, X_test = X_remaining.iloc[validation_index], X_remaining.iloc[test_index]
        y_valid, y_test = y_remaining.iloc[validation_index], y_remaining.iloc[test_index]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Trains the random forest regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Target data.

    Returns:
        Trained model.
    """
    regressor = RandomForestRegressor(criterion='squared_error',n_estimators=50,n_jobs=-1)
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: RandomForestRegressor, X_valid: pd.DataFrame, y_valid: pd.Series
) -> Dict[str, float]:
    """Calculates and logs the metrics.

    Args:
        regressor: Trained model.
        X_valid: Validating data of independent features.
        y_valid: validation taget data.
    """
    y_pred = regressor.predict(X_valid)
    score = r2_score(y_valid, y_pred)
    mae = mean_absolute_error(y_valid, y_pred)
    me = max_error(y_valid, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on validation data.", score)
    return {"r2_score": score, "mae": mae, "max_error": me}
