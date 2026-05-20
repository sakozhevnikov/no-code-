"""Автоматический подбор гиперпараметров с помощью Optuna."""
import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, make_scorer, f1_score, mean_squared_error
from ..models.model_factory import ModelFactory

def run_optuna_study(task: str, model_name: str, X, y=None, n_trials: int = 10):
    """
    Запускает исследование Optuna и возвращает лучшие параметры и значение целевой метрики.

    Args:
        task: 'Классификация', 'Регрессия' или 'Кластеризация'.
        model_name: отображаемое имя модели (например, 'RandomForestClassifier').
        X: признаки.
        y: целевая переменная (для кластеризации None).
        n_trials: количество испытаний.

    Returns:
        tuple: (best_params, best_score)
    """
    def objective(trial):
        params = {}
        # Общие гиперпараметры для древовидных моделей
        if model_name in ("RandomForestClassifier", "RandomForestRegressor",
                          "GradientBoostingClassifier", "GradientBoostingRegressor"):
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, 150)
            params['max_depth'] = trial.suggest_int('max_depth', 2, 15)
        elif model_name == "KMeans":
            params['n_clusters'] = trial.suggest_int('n_clusters', 2, 15)
        elif model_name == "DBSCAN":
            params['eps'] = trial.suggest_float('eps', 0.1, 5.0)
            params['min_samples'] = trial.suggest_int('min_samples', 2, 15)

        model = ModelFactory.get_model(model_name, **params)

        if task == "Кластеризация":
            model.fit(X)
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
            else:
                score = -1.0
            return score

        # Классификация / Регрессия — кросс-валидация на 3 фолдах
        if task == "Классификация":
            scorer = make_scorer(f1_score, average='weighted')
        elif task == "Регрессия":
            scorer = make_scorer(mean_squared_error, greater_is_better=False)  # negative MSE
        else:
            raise ValueError(f"Неизвестная задача: {task}")

        scores = cross_val_score(model, X, y, cv=3, scoring=scorer)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=1)  # n_jobs=1 для стабильности в Streamlit
    return study.best_params, study.best_value