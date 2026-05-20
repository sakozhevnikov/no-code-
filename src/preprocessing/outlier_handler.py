"""
Обработчик выбросов (Z-score). Возвращает (DataFrame, int).
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple
from .base_handler import BasePreprocessingHandler

class OutlierHandler(BasePreprocessingHandler):
    """Обрабатывает выбросы: удаление или замена средним/медианой."""

    def process(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, int]:
        column = kwargs.get("column")
        threshold = kwargs.get("threshold", 3.0)
        action = kwargs.get("action", "delete")  # 'delete', 'mean', 'median'
        return self._handle_outliers_zscore(df, column, threshold, action)

    @staticmethod
    def _handle_outliers_zscore(df: pd.DataFrame, column: str,
                                threshold: float, action: str) -> Tuple[pd.DataFrame, int]:
        df_copy = df.copy()
        col_data = df_copy[column]
        z = np.abs(stats.zscore(col_data.dropna()))
        mask = pd.Series(False, index=df_copy.index)
        mask[col_data.notna()] = z >= threshold
        num_outliers = mask.sum()

        if action == 'delete':
            return df_copy[~mask], num_outliers
        elif action == 'mean':
            replacement = col_data.mean()
        elif action == 'median':
            replacement = col_data.median()
        else:
            raise ValueError("action must be 'delete', 'mean', or 'median'")

        df_copy.loc[mask, column] = replacement
        return df_copy, num_outliers