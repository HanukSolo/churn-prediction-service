#  Churn Prediction System

Production-ready ML-сервис для прогнозирования оттока клиентов с использованием ансамбля моделей (XGBoost + Logistic Regression), развернутый в Kubernetes с полным циклом мониторинга (Prometheus + Grafana).

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-009688.svg)](https://xgboost.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Minikube-326ce5.svg)](https://kubernetes.io/)

---

## 📋 Описание

Система предсказывает вероятность оттока клиентов на основе их исторических данных (демография, поведение, финансовые показатели). Помогает бизнесу proactively удерживать ценных клиентов через targeted retention-кампании.

Проект демонстрирует полный жизненный цикл ML-модели для табличных данных: от EDA и feature engineering до контейнеризации, оркестрации в Kubernetes и настройки observability.

### 🌟 Ключевые особенности
- **Высокое качество предсказания**: ROC-AUC 0.85+ благодаря ансамблю моделей и тщательному feature engineering.
- **Модульная архитектура**: Чистый код, разделенный на `src/` (config, data, features, model, train) по принципам SOLID.
- **Production-ready API**: FastAPI с валидацией входных данных, Swagger UI и асинхронной обработкой.
- **Оркестрация**: Деплой в Kubernetes с 2 репликами, Liveness/Readiness пробами и ограничением ресурсов.
- **Мониторинг**: Интеграция с Prometheus (кастомные метрики RPS, Latency, Error Rate) и визуализация в Grafana.

---

## 🏗️ Архитектура системы

### Схема взаимодействия компонентов
````
Клиент (Swagger UI / CRM)
│
▼ HTTP POST /predict
K8s Service (LoadBalancer)
│
├──▶ Pod 1: FastAPI + XGBoost ──┐
│ │
└──▶ Pod 2: FastAPI + XGBoost ──┤
│
▼
📊 Prometheus
(scrape /metrics)
│
▼
📈 Grafana Dashboard
````

### Компоненты системы

| Компонент | Технология | Назначение |
| --- | --- | --- |
| **API-сервер** | FastAPI + Uvicorn | REST API для предсказания оттока |
| **ML-модель** | XGBoost + Logistic Regression | Ансамбль для классификации оттока |
| **Оркестратор** | Kubernetes (Minikube) | Управление подами, масштабирование, health-checks |
| **Контейнеризация** | Docker | Изоляция зависимостей и воспроизводимость |
| **Мониторинг** | Prometheus + Grafana | Сбор метрик и визуализация |
| **ServiceMonitor** | Custom Resource | Автоматическое обнаружение сервиса для Prometheus |

---

## 🛠️ Технологический стек

| Категория | Технологии |
| --- | --- |
| **ML / Data Science** | scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Инфраструктура** | Docker, Kubernetes (Minikube), Helm |
| **Мониторинг** | Prometheus, Grafana, Prometheus-Client |
| **Инструменты** | Git, PyCharm, Jupyter Notebook |

---

## 📂 Структура проекта

```text
churn-prediction-system/
├── api/                    # FastAPI приложение
│   ├── main.py             # Точки входа и эндпоинты
│   ├── dependencies.py     # Загрузка моделей (singleton)
│   └── schemas.py          # Pydantic модели запросов/ответов
── src/                    # Модули машинного обучения
│   ├── config.py           # Централизованная конфигурация
│   ├── data.py             # Загрузка и предобработка данных
│   ├── features.py         # Feature engineering
│   ├── model.py            # Построение и обучение моделей
│   ├── train.py            # Цикл обучения и валидации
│   └── evaluate.py         # Оценка метрик и визуализация
├── k8s/                    # Kubernetes манифесты
│   ├── deployment.yaml     # Деплой с 2 репликами и probes
│   ├── service.yaml        # Внутренний балансировщик
│   └── servicemonitor.yaml # Конфигурация сбора метрик Prometheus
├── data/                   # Данные и артефакты
│   ├── raw/                # Исходный датасет
│   ├── processed/          # Обработанные данные
│   └── models/             # Сохраненные модели (.pkl, .json)
├── notebooks/              # Jupyter ноутбуки для EDA
│   └── eda.ipynb
├── Dockerfile              # Инструкции для сборки образа
├── requirements.txt        # Зависимости Python
└── README.md               # Этот файл
```
## Быстрый старт
### 1. Локальный запуск (Docker)
```bash
docker build -t churn-prediction-api:latest .
docker run -p 8000:8000 churn-prediction-api:latest
```
Swagger UI: http://127.0.0.1:8000/docs
### 2. Деплой в Kubernetes (Minikube)
```bash
minikube start --driver=docker
minikube image load churn-prediction-api:latest

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install my-monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

kubectl apply -f k8s/

minikube service churn-prediction-service
minikube service my-monitoring-grafana -n monitoring
```
##  Мониторинг и Метрики

Система экспортирует кастомные метрики для оценки здоровья ML-сервиса в реальном времени через эндпоинт `/metrics`:

| Метрика | Тип | Описание |
|---------|-----|----------|
| `http_requests_total` | Counter | Общее количество запросов (с разбивкой по методу, эндпоинту и статус-коду) |
| `http_request_duration_seconds` | Histogram | Гистограмма задержки инференса (P50, P90, P99) |

### Сбор метрик

- **ServiceMonitor** автоматически обнаруживает сервис через label `app: churn-prediction`
- Prometheus скрейпит метрики каждые **15 секунд**
- Grafana визуализирует данные в реальном времени

### Ключевые дашборды

#### Requests Per Second (RPS)

```promql
sum(rate(http_requests_total{exported_endpoint="/predict"}[5m]))
```

#### Average Prediction Latency
```promql
sum(rate(http_request_duration_seconds_sum{exported_endpoint="/predict"}[5m])) 
/ 
sum(rate(http_request_duration_seconds_count{exported_endpoint="/predict"}[5m]))
```

#### Error Rate (%)
```promql
sum(rate(http_requests_total{http_status=~"4..|5..", exported_endpoint="/predict"}[5m])) 
/ 
sum(rate(http_requests_total{exported_endpoint="/predict"}[5m])) 
* 100
```

#### Total Requests
```promql
sum(http_requests_total{exported_endpoint="/predict"})
```
#### Примеры метрик в действии
- Средняя задержка предсказания: ~15 мс на запрос (CPU)
- Обработано запросов в тесте: 100 запросов без ошибок
- Error Rate: 0% (все запросы вернули статус 200)

## 📈 Результаты модели

- Алгоритмы: XGBoost + Logistic Regression (ансамбль через VotingClassifier)
- Dataset: Telco Customer Churn (~7000 клиентов, 21 признак)
- Метрики на тестовой выборке:
- ROC-AUC: 0.85+
- Precision: 0.78
- Recall: 0.72
- F1-Score: 0.75
- Средняя задержка инференса (CPU): ~15 мс на запрос
- Масштабирование: 2 реплики в Kubernetes для отказоустойчивости

## 👤 Автор
**Арсений Гирин**
[ML Engineer]
https://github.com/HanukSolo