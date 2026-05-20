"""
Фабрика моделей машинного обучения. Создаёт модель по строковому ключу.
"""
from .model_registry import MODEL_REGISTRY

class ModelFactory:
    """Фабричный класс для создания scikit-learn моделей."""

    # Маппинг отображаемых имён в ключи реестра
    DISPLAY_NAME_MAP = {
        "RandomForestClassifier": "random_forest_classifier",
        "GradientBoostingClassifier": "gradient_boosting_classifier",
        "RandomForestRegressor": "random_forest_regressor",
        "GradientBoostingRegressor": "gradient_boosting_regressor",
        "KMeans": "kmeans",
        "DBSCAN": "dbscan",
    }

    @staticmethod
    def get_model(name: str, **kwargs):
        """
        Возвращает экземпляр модели с переданными гиперпараметрами.

        Args:
            name: строковое имя модели (ключ из MODEL_REGISTRY или отображаемое имя).
            **kwargs: гиперпараметры модели.

        Returns:
            Объект модели sklearn.

        Raises:
            ValueError: если модель не найдена.
        """
        # Сначала проверяем прямое совпадение с ключом реестра
        if name in MODEL_REGISTRY:
            model_cls = MODEL_REGISTRY[name]
        else:
            # Пробуем преобразовать через маппинг отображаемых имён
            mapped = ModelFactory.DISPLAY_NAME_MAP.get(name)
            if mapped and mapped in MODEL_REGISTRY:
                model_cls = MODEL_REGISTRY[mapped]
            else:
                raise ValueError(
                    f"Модель '{name}' не поддерживается. Доступные: {list(ModelFactory.DISPLAY_NAME_MAP.keys())}"
                )
        return model_cls(**kwargs)