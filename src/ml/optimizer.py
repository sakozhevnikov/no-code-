"""
Подбор гиперпараметров с помощью Optuna для всех типов задач.
"""
import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, mean_squared_error, silhouette_score
from ..models.model_factory import ModelFactory
from src.config import OPTUNA_N_TRIALS, OPTUNA_RANDOM_STATE

def run_optuna_study(task: str, model_name: str, X, y=None, n_trials: int = OPTUNA_N_TRIALS):
    def objective(trial):
        params = {}
        if model_name in ("RandomForestClassifier", "RandomForestRegressor",
                          "GradientBoostingClassifier", "GradientBoostingRegressor"):
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 2, 20)
        elif model_name == "KMeans":
            params['n_clusters'] = trial.suggest_int('n_clusters', 2, 20)
        elif model_name == "DBSCAN":
            params['eps'] = trial.suggest_float('eps', 0.1, 5.0)
            params['min_samples'] = trial.suggest_int('min_samples', 2, 20)

        model = ModelFactory.get_model(model_name, **params)

        if task == "Кластеризация":
            model.fit(X)
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
            else:
                score = -1.0
            return score

        if task == "Классификация":
            scorer = make_scorer(f1_score, average='weighted')
        elif task == "Регрессия":
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
        else:
            raise ValueError(f"Неизвестная задача: {task}")

        scores = cross_val_score(model, X, y, cv=3, scoring=scorer)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        seed=OPTUNA_RANDOM_STATE
    )
    return study.best_params, study.best_value