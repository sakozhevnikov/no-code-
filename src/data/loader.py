"""
Модуль загрузки табличных данных.
Поддерживает CSV и XLSX.
"""
import pandas as pd
from typing import Tuple

def load_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Загружает DataFrame из объекта UploadedFile библиотеки Streamlit.

    Args:
        uploaded_file: объект, возвращаемый st.file_uploader.

    Returns:
        tuple: (DataFrame, тип_файла).
    """
    filename = uploaded_file.name.lower()
    if filename.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        file_type = 'csv'
    elif filename.endswith('.xlsx'):
        # openpyxl – стандартный движок для xlsx
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        file_type = 'xlsx'
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {filename}")

    return df, file_type