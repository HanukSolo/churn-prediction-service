"""
Модуль для подготовки данных и создания препроцессора.
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple

from src.config import RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]


def get_preprocessor() -> ColumnTransformer:
    """
    Создает и возвращает объект ColumnTransformer для предобработки признаков.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def prepare_data(
        df: pd.DataFrame,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    logger.info("Начало подготовки данных (разделение X и y, train/test split)")

    drop_cols = ['customerID', 'Churn']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['Churn'].map({'No': 0, 'Yes': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Данные разделены. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    logger.info(f"Доля оттока в train: {y_train.mean():.2%}, в test: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test