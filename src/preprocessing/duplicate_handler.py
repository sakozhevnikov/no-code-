"""
Обработчик дубликатов.
"""
import pandas as pd
from .base_handler import BasePreprocessingHandler

class DuplicateHandler(BasePreprocessingHandler):
    """Удаляет дубликаты строк."""

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        subset = kwargs.get("subset", None)
        keep = kwargs.get("keep", 'first')
        return self._remove_duplicates(df, subset, keep)

    @staticmethod
    def _remove_duplicates(df: pd.DataFrame, subset=None, keep='first') -> pd.DataFrame:
        return df.drop_duplicates(subset=subset, keep=keep)