"""
Модуль для загрузки и первичной очистки данных.
Отвечает только за чтение CSV и базовую валидацию.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

from src.config import RAW_DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_clean_data(file_path: Optional[str] = None) -> pd.DataFrame:

    if file_path is None:
        file_path = RAW_DATA_PATH

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    logger.info(f"Загрузка данных из {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Загруженный файл пустой")

    required_columns = ['customerID', 'TotalCharges', 'Churn']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

    logger.info(f"Исходный размер данных: {df.shape}")

    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    null_count = df['TotalCharges'].isnull().sum()
    if null_count > 0:
        logger.warning(f"Осталось {null_count} пропусков в TotalCharges после очистки")

    logger.info(f"Данные успешно загружены и очищены. Итоговый размер: {df.shape}")

    return df


if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.head())
    print(f"\nТип TotalCharges: {df['TotalCharges'].dtype}")
    print(f"Пропуски в TotalCharges: {df['TotalCharges'].isnull().sum()}")