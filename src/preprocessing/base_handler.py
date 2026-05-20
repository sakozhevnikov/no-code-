"""
Базовый абстрактный класс для всех обработчиков предобработки.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class BasePreprocessingHandler(ABC):
    """Единый интерфейс обработчика данных."""

    @abstractmethod
    def process(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Применяет предобработку к DataFrame.

        Аргументы зависят от конкретного обработчика.
        Возвращает обработанный DataFrame.
        """
        pass