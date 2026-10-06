"""
Главный файл FastAPI приложения.
"""
import logging
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pathlib import Path

from api.schemas import CustomerData
from src.config import PROCESSED_DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_pipeline = None

EXPECTED_COLUMNS = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'SeniorCitizen',
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Загружает модель в память при старте приложения и освобождает при остановке.
    """
    global model_pipeline
    model_path = PROCESSED_DATA_PATH / "baseline_model.pkl"

    if not model_path.exists():
        logger.error(f"Модель не найдена по пути: {model_path}. Сначала запустите train.py")
        raise FileNotFoundError("Модель не найдена")

    logger.info(f"Загрузка модели из {model_path}...")
    model_pipeline = joblib.load(model_path)
    logger.info("Модель успешно загружена в память!")

    yield

    logger.info("Очистка ресурсов при завершении работы приложения...")
    model_pipeline = None

app = FastAPI(
    title="Churn Prediction API",
    description="ML сервис для предсказания оттока клиентов телеком-компании",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    """
    Эндпоинт для проверки жизнеспособности сервиса.
    """
    return {"status": "healthy", "model_loaded": model_pipeline is not None}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    """
    Принимает данные клиента и возвращает вероятность оттока.
    """
    try:
        data_dict = customer.model_dump()

        default_values = {
            'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'PhoneService': 'Yes', 'MultipleLines': 'No', 'OnlineSecurity': 'No',
            'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
            'StreamingTV': 'No', 'StreamingMovies': 'No', 'PaperlessBilling': 'Yes'
        }

        for col, default_val in default_values.items():
            if col not in data_dict:
                data_dict[col] = default_val

        df_input = pd.DataFrame([data_dict])[EXPECTED_COLUMNS]

        probabilities = model_pipeline.predict_proba(df_input)[0]
        churn_probability = float(probabilities[1])  # Берем вероятность класса 1 (Yes)

        return {
            "churn_probability": round(churn_probability, 4),
            "prediction": "Yes" if churn_probability >= 0.5 else "No",
            "message": "Клиент склонен к оттоку" if churn_probability >= 0.5 else "Клиент лоялен"
        }

    except Exception as e:
        logger.error(f"Ошибка при предсказании: {str(e)}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при обработке данных")