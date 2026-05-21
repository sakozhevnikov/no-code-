"""
Модуль загрузки табличных данных.
Поддерживаемые форматы берутся из конфигурации.
"""
import pandas as pd
from typing import Tuple
from src.config import SUPPORTED_FILE_TYPES   # <-- добавлен импорт

def load_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    filename = uploaded_file.name.lower()
    if filename.endswith('.csv') and 'csv' in SUPPORTED_FILE_TYPES:
        df = pd.read_csv(uploaded_file)
        file_type = 'csv'
    elif filename.endswith('.xlsx') and 'xlsx' in SUPPORTED_FILE_TYPES:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        file_type = 'xlsx'
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {filename}")
    return df, file_type