"""
Страница машинного обучения: классификация, регрессия, кластеризация.
С автоматическим подбором гиперпараметров через Optuna.
"""
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from ...models.model_factory import ModelFactory
from ...ml.training import train_model
from ...ml.metrics import evaluate_classification, evaluate_regression, evaluate_clustering
from ...ml.optimizer import run_optuna_study
from ..components.notifications import show_success, show_error, show_info, show_warning
from ..components.charts import (
    show_confusion_matrix, show_boxplot, show_prediction_plot, show_cluster_plot
)
from ..components.metrics import show_classification_metrics, show_regression_metrics, show_clustering_metrics
from ..utils.session import get_data, set_data, set_model, get_model
from ..utils.validators import (
    check_data_loaded, check_target_selected, check_features_selected,
    check_no_missing_in_target
)

def render() -> None:
    st.title("🤖 Машинное обучение")
    df = get_data()
    if not check_data_loaded(df):
        return

    # Выбор типа задачи
    task = st.selectbox("Тип задачи:", ("Классификация", "Регрессия", "Кластеризация"))
    # Сброс меток кластеров, если переключились с кластеризации на другую задачу
    if task != "Кластеризация" and 'cluster_labels' in st.session_state:
        del st.session_state['cluster_labels']
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    if task == "Кластеризация":
        target = None
        features = st.multiselect("Признаки для кластеризации:", numeric_cols)
    else:
        target = st.selectbox("Целевая переменная:", df.columns.tolist())
        features = st.multiselect("Признаки:", [c for c in df.columns if c != target])

    # Выбор модели
    if task == "Классификация":
        model_display_name = st.selectbox("Модель:", ("RandomForestClassifier", "GradientBoostingClassifier"))
    elif task == "Регрессия":
        model_display_name = st.selectbox("Модель:", ("RandomForestRegressor", "GradientBoostingRegressor"))
    else:
        model_display_name = st.selectbox("Модель:", ("KMeans", "DBSCAN"))

    # Гиперпараметры (ручной ввод)
    st.subheader("⚙️ Гиперпараметры")
    params = {}

    if model_display_name in ("RandomForestClassifier", "RandomForestRegressor",
                              "GradientBoostingClassifier", "GradientBoostingRegressor"):
        params['n_estimators'] = st.slider("n_estimators", 10, 500, 100)
        params['max_depth'] = st.slider("max_depth", 1, 50, 5)

    elif model_display_name == "KMeans":
        params['n_clusters'] = st.slider("n_clusters", 2, 20, 3)

    elif model_display_name == "DBSCAN":
        params['eps'] = st.slider("eps", 0.1, 10.0, 0.5)
        params['min_samples'] = st.slider("min_samples", 2, 20, 5)

    # Optuna
    use_optuna = st.checkbox("🔍 Использовать автоматический подбор гиперпараметров (Optuna)")

    # Кнопка обучения
    if st.button("🚀 Запустить обучение"):
        # Валидации
        if task != "Кластеризация":
            if not check_target_selected(target):
                return
            if not check_no_missing_in_target(df, target):
                return
        if not check_features_selected(features):
            return

        try:
            # Подготовка данных
            X = df[features].copy()
            if task != "Кластеризация":
                y = df[target].copy()
                if task == "Классификация" and not pd.api.types.is_numeric_dtype(y):
                    from sklearn.preprocessing import LabelEncoder
                    y = LabelEncoder().fit_transform(y)
            else:
                y = None

            # Разделение train/test для контролируемых задач
            if task != "Кластеризация":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = X, None, None, None

            # ---- Optuna ----
            if use_optuna:
                with st.spinner("Optuna подбирает гиперпараметры..."):
                    best_params, best_score = run_optuna_study(
                        task=task,
                        model_name=model_display_name,
                        X=X_train,
                        y=y_train,
                        n_trials=10
                    )
                show_success(f"Optuna завершена. Лучший score: {best_score:.4f}")
                st.json(best_params)
                params = best_params
            # ------------------

            # Создание и обучение модели
            model = ModelFactory.get_model(model_display_name, **params)
            if task == "Кластеризация":
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)

            set_model(model, model_display_name)

            # Предсказание и метрики
            if task == "Кластеризация":
                labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_train)
                metrics = evaluate_clustering(X_train, labels)
                show_clustering_metrics(metrics)

                # Если метрика N/A – предупреждение
                if metrics.get("Silhouette Score") is None:
                    show_warning(
                        "Silhouette Score недоступен: модель выделила 0 или 1 кластер. "
                        "Попробуйте изменить параметры (для DBSCAN – увеличьте `eps` или уменьшите `min_samples`)."
                    )

                show_cluster_plot(X_train, labels)

                # Распределение по кластерам
                import numpy as np
                unique, counts = np.unique(labels, return_counts=True)
                st.subheader("📊 Количество объектов в каждом кластере")
                cluster_names = ["Шум (-1)" if u == -1 else f"Кластер {u}" for u in unique]
                st.table({"Кластер": cluster_names, "Количество": counts})

                # Сохраняем метки для последующего добавления
                st.session_state['cluster_labels'] = labels
            else:
                y_pred = model.predict(X_test)
                if task == "Классификация":
                    metrics = evaluate_classification(y_test, y_pred)
                    show_classification_metrics(metrics)
                    show_confusion_matrix(y_test, y_pred)
                else:  # Регрессия
                    metrics = evaluate_regression(y_test, y_pred)
                    show_regression_metrics(metrics)
                    show_prediction_plot(y_test, y_pred)

        except Exception as e:
            show_error(f"Ошибка обучения: {e}")

    # Кнопка для добавления меток кластеров (вне основного блока)
    if task == "Кластеризация" and 'cluster_labels' in st.session_state:
        if st.button("💾 Добавить метки кластеров в данные"):
            df = get_data()
            df['cluster'] = st.session_state['cluster_labels'].astype(int)
            set_data(df)
            show_success("Столбец 'cluster' добавлен. Можно использовать в анализе.")
            del st.session_state['cluster_labels']

    # Просмотр текущей модели
    if st.button("📋 Показать текущую модель"):
        model, name = get_model()
        if model:
            st.write(f"Обученная модель: **{name}**")
        else:
            show_info("Модель ещё не обучена.")