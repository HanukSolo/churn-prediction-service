"""
Streamlit-приложение для демонстрации ML-сервиса предсказания оттока клиентов.
"""
import streamlit as st
import pandas as pd
import requests
import json

# Настройка страницы
st.set_page_config(
    page_title="Churn Prediction Demo",
    page_icon="📊",
    layout="wide"
)

# Заголовок
st.title(" Churn Prediction Service")
st.markdown("""
**ML-сервис для предсказания оттока клиентов телеком-компании**

Этот демо показывает работу production-ready ML-пайплайна с:
- ✅ FastAPI REST API
- ✅ Docker-контейнеризация
- ✅ MLOps (DVC + MLflow)
- ✅ Валидация данных через Pydantic
""")

# Сайдбар с информацией
st.sidebar.header(" О проекте")
st.sidebar.markdown("""
**Технологии:**
- Python, Scikit-learn
- FastAPI, Pydantic
- Docker, MLflow, DVC
- Streamlit (это демо)

**Метрики модели:**
- ROC-AUC: 0.842
- Recall: 0.559
- Precision: 0.659

**Бизнес-ценность:**
Модель помогает выявить клиентов с высоким риском оттока, 
что позволяет компании предложить им специальные условия 
и сохранить до 1.5 млн рублей на каждой выборке.
""")

# Форма для ввода данных клиента
st.header(" Предсказание оттока клиента")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Срок обслуживания (месяцев)", min_value=0, max_value=72, value=12, step=1)
    monthly_charges = st.number_input("Ежемесячная плата ($)", min_value=0.0, max_value=120.0, value=50.0, step=0.01)
    total_charges = st.number_input("Общая сумма платежей ($)", min_value=0.0, max_value=9000.0, value=600.0, step=0.01)

with col2:
    contract = st.selectbox("Тип контракта", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Тип интернета", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Способ оплаты", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

# Кнопка предсказания
st.markdown("---")
if st.button("🚀 Предсказать отток", type="primary", use_container_width=True):

    # Формируем запрос
    customer_data = {
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method
    }

    # Показываем введенные данные
    st.subheader("📋 Данные клиента")
    st.json(customer_data)

    # Отправляем запрос к API
    try:
        # Пробуем подключиться к FastAPI (если запущен)
        response = requests.post(
            "http://localhost:8000/predict",
            json=customer_data,
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            churn_prob = result["churn_probability"]
            prediction = result["prediction"]
            message = result["message"]
        else:
            # Если API не запущен, используем локальную модель
            st.warning("⚠️ API не доступен. Используем локальную модель...")
            from src.data_loader import load_and_clean_data
            from src.preprocessing import prepare_data, get_preprocessor
            from src.model import build_pipeline
            import joblib
            from pathlib import Path

            # Загружаем модель
            model_path = Path("data/processed/baseline_model.pkl")
            if model_path.exists():
                pipeline = joblib.load(model_path)

                # Создаем DataFrame с дефолтными значениями
                default_data = {
                    'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
                    'PhoneService': 'Yes', 'MultipleLines': 'No', 'OnlineSecurity': 'No',
                    'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
                    'StreamingTV': 'No', 'StreamingMovies': 'No', 'PaperlessBilling': 'Yes'
                }
                default_data.update(customer_data)

                df_input = pd.DataFrame([default_data])
                proba = pipeline.predict_proba(df_input)[0]
                churn_prob = float(proba[1])
                prediction = "Yes" if churn_prob >= 0.5 else "No"
                message = "Клиент склонен к оттоку" if churn_prob >= 0.5 else "Клиент лоялен"
            else:
                st.error("❌ Модель не найдена. Запустите `python train.py` сначала.")
                st.stop()

        # Визуализация результата
        st.subheader("📊 Результат предсказания")

        # Прогресс-бар вероятности
        st.metric("Вероятность оттока", f"{churn_prob:.2%}")

        # Цветной индикатор
        if churn_prob >= 0.7:
            st.error(f"🔴 **ВЫСОКИЙ РИСК**: {message}")
        elif churn_prob >= 0.5:
            st.warning(f"🟡 **СРЕДНИЙ РИСК**: {message}")
        else:
            st.success(f"🟢 **НИЗКИЙ РИСК**: {message}")

        # Визуализация
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Вероятность оттока", f"{churn_prob:.1%}")
        with col2:
            st.metric("Предсказание", prediction)
        with col3:
            risk_level = "Высокий" if churn_prob >= 0.7 else "Средний" if churn_prob >= 0.5 else "Низкий"
            st.metric("Уровень риска", risk_level)

        # График вероятности
        st.markdown("---")
        st.subheader(" Визуализация")

        # Создаем простой бар-чарт
        chart_data = pd.DataFrame({
            'Категория': ['Вероятность оттока', 'Вероятность сохранения'],
            'Значение': [churn_prob, 1 - churn_prob]
        })

        st.bar_chart(chart_data.set_index('Категория'))

        # Рекомендации для бизнеса
        st.markdown("---")
        st.subheader("💡 Рекомендации для бизнеса")

        if churn_prob >= 0.7:
            st.markdown("""
            - 🎯 **Срочно связаться с клиентом**
            - 💰 Предложить персональную скидку 20-30%
            - 📞 Назначить персонального менеджера
            - 🎁 Предложить бонусы или дополнительные услуги
            """)
        elif churn_prob >= 0.5:
            st.markdown("""
            - 📧 Отправить email с специальным предложением
            - 📊 Проанализировать историю использования услуг
            -  Предложить переход на долгосрочный контракт
            """)
        else:
            st.markdown("""
            - ✅ Клиент лоялен, стандартное обслуживание
            - 📈 Можно предложить дополнительные услуги (upsell)
            - ⭐ Попросить оставить отзыв или рекомендацию
            """)

    except requests.exceptions.ConnectionError:
        st.error(" Не удалось подключиться к API. Убедитесь, что FastAPI запущен: `uvicorn api.main:app --reload`")
    except Exception as e:
        st.error(f" Ошибка: {str(e)}")

# Футер
st.markdown("---")
st.markdown("""
**🔗 Ссылки на проект:**
- [GitHub Repository](https://github.com/ТВОЙ_ЛОГИН/churn-prediction-service)
- [MLflow UI](http://localhost:5000)
- [FastAPI Docs](http://localhost:8000/docs)

** Контакты:**
- Email: a-girin@bk.ru
- Github: https://github.com/HanukSolo
""")