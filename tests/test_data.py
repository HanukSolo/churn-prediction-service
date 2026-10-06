"""
Unit-тесты для модуля data_loader.
"""
import pandas as pd
import pytest
from src.data_loader import load_and_clean_data
from src.config import RAW_DATA_PATH


def test_load_and_clean_data_returns_dataframe():
    """
    Тест проверяет, что функция возвращает объект pandas DataFrame
    и размер данных соответствует ожидаемому.
    """
    df = load_and_clean_data(RAW_DATA_PATH)

    assert isinstance(df, pd.DataFrame), "Функция должна возвращать pandas DataFrame"

    assert len(df) == 7043, f"Ожидалось 7043 строки, получено {len(df)}"


def test_total_charges_is_cleaned_correctly():
    """
    Тест проверяет, что 'ловушка' TotalCharges была корректно обработана:
    1. Тип данных должен быть float64
    2. Не должно быть пропусков (NaN)
    """
    df = load_and_clean_data(RAW_DATA_PATH)

    assert df['TotalCharges'].dtype == 'float64', f"Ожидался float64, получен {df['TotalCharges'].dtype}"

    null_count = df['TotalCharges'].isnull().sum()
    assert null_count == 0, f"В TotalCharges остались пропуски: {null_count}"


def test_missing_file_raises_error():
    """
    Тест проверяет, что функция корректно выбрасывает ошибку,
    если файл не найден (защита от сбоев в продакшене).
    """
    with pytest.raises(FileNotFoundError):
        load_and_clean_data("non_existent_file.csv")