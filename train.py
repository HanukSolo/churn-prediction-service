"""
Главный скрипт для запуска обучения с MLflow и DVC.
"""
import logging
import json
from pathlib import Path
from src.data_loader import load_and_clean_data
from src.preprocessing import prepare_data
from src.model import build_pipeline, train_and_evaluate
from src.config import PROCESSED_DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== ЗАПУСК ПАЙПЛАЙНА ОБУЧЕНИЯ (DVC + MLflow) ===")

    # 1. Загружаем данные
    df = load_and_clean_data()

    X_train, X_test, y_train, y_test = prepare_data(df)

    pipeline = build_pipeline()
    metrics = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test, run_name="logreg_baseline_v1")

    metrics_path = PROCESSED_DATA_PATH / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Метрики сохранены в {metrics_path}")

    logger.info("=== ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО ===")

if __name__ == "__main__":
    main()