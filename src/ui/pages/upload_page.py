"""
Страница загрузки данных.
"""
import streamlit as st
from ...data.loader import load_uploaded_file
from ..components.notifications import show_success, show_error, show_info
from ..utils.session import set_data, clear_data, set_file_type

def render() -> None:
    st.title("📂 Загрузка данных")
    st.markdown("Загрузите файл формата CSV или XLSX для начала работы.")
    uploaded_file = st.file_uploader("Выберите файл", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            df, file_type = load_uploaded_file(uploaded_file)
            set_data(df)
            set_file_type(file_type)
            show_success(f"Файл `{uploaded_file.name}` успешно загружен (тип: {file_type}).")

            st.subheader("📋 Краткая информация")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Строк", df.shape[0])
            with col2:
                st.metric("Столбцов", df.shape[1])
        except Exception as e:
            show_error(f"Ошибка обработки файла: {e}")
    else:
        clear_data()
        show_info("Ожидание загрузки файла...")