"""
Модуль для обучения и оценки моделей с трекингом MLflow.
"""
import logging
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd

from src.config import RANDOM_STATE, PROCESSED_DATA_PATH
from src.preprocessing import get_preprocessor

logger = logging.getLogger(__name__)

def build_pipeline() -> Pipeline:
    preprocessor = get_preprocessor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])
    return pipeline

def train_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    run_name: str = "baseline_logreg"
) -> dict:
    """
    Обучает пайплайн и логирует всё в MLflow.
    """
    logger.info(f"Начало обучения модели (MLflow run: {run_name})...")

    with mlflow.start_run(run_name=run_name):
        # 1. Логируем параметры
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", RANDOM_STATE)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        report_dict = classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)'], output_dict=True)
        recall_churn = report_dict['Churn (1)']['recall']
        precision_churn = report_dict['Churn (1)']['precision']

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("recall_churn", recall_churn)
        mlflow.log_metric("precision_churn", precision_churn)

        # 6. Логируем модель как артефакт
        mlflow.sklearn.log_model(pipeline, "model")

        # 7. Также сохраняем локально для нашего API
        save_path = PROCESSED_DATA_PATH / "baseline_model.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)

        logger.info(f"Метрики: ROC-AUC={roc_auc:.4f}, Recall={recall_churn:.4f}")
        logger.info(f"Модель сохранена в {save_path} и залогирована в MLflow")

        return {"roc_auc": roc_auc, "recall": recall_churn}