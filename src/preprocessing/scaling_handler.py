"""
Масштабирование числовых данных.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .base_handler import BasePreprocessingHandler

class ScalingHandler(BasePreprocessingHandler):
    """Масштабирует числовые столбцы."""

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        method = kwargs.get("method", "standard")  # 'standard' или 'minmax'
        columns = kwargs.get("columns", [])
        if method == "standard":
            return self._standard_scale(df, columns)
        elif method == "minmax":
            return self._minmax_scale(df, columns)
        else:
            raise ValueError(f"Неизвестный метод масштабирования: {method}")

    @staticmethod
    def _standard_scale(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df_copy = df.copy()
        scaler = StandardScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns].astype(float))
        return df_copy

    @staticmethod
    def _minmax_scale(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df_copy = df.copy()
        scaler = MinMaxScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns].astype(float))
        return df_copy