"""
Обработчик категориальных признаков (Label / One‑Hot Encoding).
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .base_handler import BasePreprocessingHandler

class CategoricalHandler(BasePreprocessingHandler):
    """Кодирует категориальные столбцы."""

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        encoding = kwargs.get("encoding", "label")  # 'label' или 'onehot'
        columns = kwargs.get("columns", [])
        if encoding == "label":
            return self._label_encode(df, columns)
        elif encoding == "onehot":
            return self._onehot_encode(df, columns)
        else:
            raise ValueError(f"Неизвестный тип кодирования: {encoding}")

    @staticmethod
    def _label_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df_copy = df.copy()
        le = LabelEncoder()
        for col in columns:
            df_copy[col] = df_copy[col].fillna('__missing__')
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        return df_copy

    @staticmethod
    def _onehot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        return pd.get_dummies(df, columns=columns, drop_first=False)