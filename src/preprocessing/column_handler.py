"""
Обработчик удаления произвольных столбцов.
"""
import pandas as pd
from .base_handler import BasePreprocessingHandler

class ColumnHandler(BasePreprocessingHandler):
    """Удаляет выбранные столбцы."""

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        columns = kwargs.get("columns", [])
        return self._delete_columns(df, columns)

    @staticmethod
    def _delete_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        return df.drop(columns=columns, errors='ignore')