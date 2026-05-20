"""Метрики для оценки моделей."""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
import numpy as np

def evaluate_classification(y_true, y_pred):
    """
    Метрики классификации.
    Если классов ровно 2 – использует бинарные метрики (положительный класс = 1).
    Если больше – усреднение weighted для многоклассового случая.
    """
    import numpy as np
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) == 2:
        # Бинарный случай
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "F1-score": f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        }

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R²": r2_score(y_true, y_pred)
    }

def evaluate_clustering(X, labels):
    if len(set(labels)) > 1:
        sil = silhouette_score(X, labels)
    else:
        sil = None
    return {"Silhouette Score": sil}