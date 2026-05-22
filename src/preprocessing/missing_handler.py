"""
Обработчик пропущенных значений (наследует BasePreprocessingHandler).
Поддерживает удаление строк/столбцов по порогу пропусков и заполнение пропусков.
"""
import pandas as pd
from .base_handler import BasePreprocessingHandler

class MissingHandler(BasePreprocessingHandler):
    """Обрабатывает пропуски."""

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        method = kwargs.get("method", "delete_rows")
        columns = kwargs.get("columns", None)
        threshold = kwargs.get("threshold", 0.0)

        if method == "delete_rows":
            return self._delete_rows(df, columns, threshold)
        elif method == "delete_columns":
            return self._delete_columns(df, threshold)
        elif method == "fill_mean":
            return self._fill_mean(df, columns)
        elif method == "fill_median":
            return self._fill_median(df, columns)
        elif method == "fill_mode":
            return self._fill_mode(df, columns)
        else:
            raise ValueError(f"Неизвестный метод обработки пропусков: {method}")

    # ---------- методы удаления ----------
    @staticmethod
    def _delete_rows(df: pd.DataFrame, columns=None, threshold=0.0) -> pd.DataFrame:
        """Удаляет строки, где доля пропусков (в указанных столбцах) >= threshold."""
        subset = columns if columns else df.columns
        missing_ratio = df[subset].isnull().mean(axis=1)
        return df[missing_ratio < threshold]

    @staticmethod
    def _delete_columns(df: pd.DataFrame, threshold=0.0) -> pd.DataFrame:
        """Удаляет столбцы, в которых доля пропусков >= threshold."""
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio >= threshold].index
        return df.drop(columns=cols_to_drop)

    # ---------- методы заполнения (без удаления столбцов) ----------
    @staticmethod
    def _fill_mean(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Заполняет пропуски средним значением в указанных столбцах."""
        df_copy = df.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        return df_copy

    @staticmethod
    def _fill_median(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Заполняет пропуски медианой в указанных столбцах."""
        df_copy = df.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        return df_copy

    @staticmethod
    def _fill_mode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Заполняет пропуски модой в указанных столбцах."""
        df_copy = df.copy()
        for col in columns:
            mode_val = df_copy[col].mode()
            if not mode_val.empty:
                df_copy[col] = df_copy[col].fillna(mode_val[0])
        return df_copy