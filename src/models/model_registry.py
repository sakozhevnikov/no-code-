"""
Реестр моделей машинного обучения (только классы, без параметров).
"""
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.cluster import KMeans, DBSCAN

MODEL_REGISTRY = {
    # Классификация
    "random_forest_classifier": RandomForestClassifier,
    "gradient_boosting_classifier": GradientBoostingClassifier,
    # Регрессия
    "random_forest_regressor": RandomForestRegressor,
    "gradient_boosting_regressor": GradientBoostingRegressor,
    # Кластеризация
    "kmeans": KMeans,
    "dbscan": DBSCAN,
}