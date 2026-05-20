"""
Фабрика обработчиков предобработки. Использует реестр для выбора обработчика.
"""
from .base_handler import BasePreprocessingHandler
from .missing_handler import MissingHandler
from .duplicate_handler import DuplicateHandler
from .outlier_handler import OutlierHandler
from .categorical_handler import CategoricalHandler
from .scaling_handler import ScalingHandler
from .column_handler import ColumnHandler

HANDLER_REGISTRY = {
    "missing": MissingHandler,
    "duplicates": DuplicateHandler,
    "outliers": OutlierHandler,
    "categorical": CategoricalHandler,
    "scaling": ScalingHandler,
    "columns": ColumnHandler,
}


class PreprocessingFactory:
    """Возвращает экземпляр обработчика по имени."""

    @staticmethod
    def get_handler(name: str) -> BasePreprocessingHandler:
        handler_cls = HANDLER_REGISTRY.get(name)
        if not handler_cls:
            raise ValueError(f"Обработчик '{name}' не найден. Доступные: {list(HANDLER_REGISTRY.keys())}")
        return handler_cls()