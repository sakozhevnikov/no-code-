"""
Функции валидации действий пользователя.
Все проверки вызывают notifications при ошибке.
"""
import pandas as pd
from ..components.notifications import show_error, show_warning

def check_data_loaded(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        show_error("Данные не загружены. Сначала загрузите файл.")
        return False
    return True

def check_target_selected(target: str) -> bool:
    if not target:
        show_warning("Выберите целевую переменную.")
        return False
    return True

def check_features_selected(features: list) -> bool:
    if not features:
        show_warning("Выберите хотя бы один признак.")
        return False
    return True

def check_numeric_columns(df: pd.DataFrame, cols: list) -> bool:
    if not cols:
        show_error("Нет числовых столбцов.")
        return False
    return True

def check_categorical_columns(df: pd.DataFrame, cols: list) -> bool:
    if not cols:
        show_error("Нет категориальных столбцов.")
        return False
    return True

def check_min_samples(df: pd.DataFrame, min_samples: int = 10) -> bool:
    if len(df) < min_samples:
        show_error(f"Недостаточно данных. Минимум {min_samples} строк.")
        return False
    return True

def check_no_missing_in_target(df: pd.DataFrame, target: str) -> bool:
    """Проверяет, что в целевой переменной нет пропусков."""
    if df[target].isnull().any():
        show_error(f"В целевой переменной '{target}' есть пропуски. Сначала обработайте их в разделе «Предобработка данных».")
        return False
    return True