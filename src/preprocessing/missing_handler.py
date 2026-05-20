"""
Обработчик пропущенных значений (наследует BasePreprocessingHandler).
"""
import pandas as pd
from .base_handler import BasePreprocessingHandler

class MissingHandler(BasePreprocessingHandler):
    """Обрабатывает пропуски: удаление строк/столбцов или заполнение."""

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        method = kwargs.get("method", "delete_rows")
        columns = kwargs.get("columns", None)
        threshold = kwargs.get("threshold", 0.0)

        if method == "delete_rows":
            return self._delete_rows(df, columns, threshold)
        elif method == "delete_columns":
            return self._delete_columns(df, threshold)
        elif method == "fill_mean":
            return self._fill_mean(df, columns, threshold)
        elif method == "fill_median":
            return self._fill_median(df, columns, threshold)
        elif method == "fill_mode":
            return self._fill_mode(df, columns, threshold)
        else:
            raise ValueError(f"Неизвестный метод обработки пропусков: {method}")

    # Внутренние статические методы (сохранены как есть, только приватные)
    @staticmethod
    def _delete_rows(df: pd.DataFrame, columns=None, threshold=0.0) -> pd.DataFrame:
        if columns is None:
            subset = df.columns
        else:
            subset = columns
        missing_ratio = df[subset].isnull().mean(axis=1)
        return df[missing_ratio < threshold]

    @staticmethod
    def _delete_columns(df: pd.DataFrame, threshold=0.0) -> pd.DataFrame:
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio >= threshold].index
        return df.drop(columns=cols_to_drop)

    @staticmethod
    def _fill_mean(df: pd.DataFrame, columns, threshold=0.0) -> pd.DataFrame:
        return MissingHandler._fill_with_strategy(df, columns, threshold, 'mean')

    @staticmethod
    def _fill_median(df: pd.DataFrame, columns, threshold=0.0) -> pd.DataFrame:
        return MissingHandler._fill_with_strategy(df, columns, threshold, 'median')

    @staticmethod
    def _fill_mode(df: pd.DataFrame, columns, threshold=0.0) -> pd.DataFrame:
        return MissingHandler._fill_with_strategy(df, columns, threshold, 'mode')

    @staticmethod
    def _fill_with_strategy(df: pd.DataFrame, columns, threshold: float, strategy: str) -> pd.DataFrame:
        df_copy = df.copy()
        cols_to_drop = []
        for col in columns:
            missing_ratio = df_copy[col].isnull().mean()
            if missing_ratio >= threshold:
                cols_to_drop.append(col)
            else:
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].fillna(df_copy[col].median())
                elif strategy == 'mode':
                    mode_val = df_copy[col].mode()
                    if not mode_val.empty:
                        df_copy[col] = df_copy[col].fillna(mode_val[0])
        if cols_to_drop:
            df_copy = df_copy.drop(columns=cols_to_drop)
        return df_copy