"""
Страница предварительного анализа данных.
"""
import streamlit as st
import pandas as pd
from ...data.eda import EDAAnalyzer
from ..components.notifications import show_info
from ..components.data_preview import show_head, show_tail, show_full
from ..utils.session import get_data
from ..utils.validators import check_data_loaded
from ..components.charts import show_histogram, show_scatter, show_correlation_heatmap

def render() -> None:
    st.title("📊 Предварительный анализ данных")
    df = get_data()
    if not check_data_loaded(df):
        return

    analyzer = EDAAnalyzer(df)

    # Просмотр данных
    st.subheader("📄 Просмотр данных")
    view_mode = st.radio(
        "Режим просмотра:",
        ("Первые N строк", "Последние N строк", "Вся таблица"),
        horizontal=True
    )
    if view_mode == "Вся таблица":
        show_full(df)
    else:
        n_rows = st.number_input("Количество строк:", min_value=1, max_value=len(df), value=5, step=1)
        if view_mode == "Первые N строк":
            show_head(df, n_rows)
        else:
            show_tail(df, n_rows)

    # Базовая информация
    analyzer.run_full_eda()

    # Графики EDA
    st.header("📈 Построение графиков")

    # Гистограмма
    with st.expander("📊 Гистограмма", expanded=False):
        if analyzer.numeric_cols:
            hist_col = st.selectbox("Числовой признак:", analyzer.numeric_cols, key="hist_col")
            if st.button("Построить гистограмму", key="hist_btn"):
                show_histogram(df, hist_col)
        else:
            show_info("Нет числовых признаков.")

    # Диаграмма рассеяния
    with st.expander("🔵 Диаграмма рассеяния", expanded=False):
        if len(analyzer.numeric_cols) >= 2:
            col_x = st.selectbox("Признак X:", analyzer.numeric_cols, key="scatter_x")
            col_y = st.selectbox("Признак Y:", [c for c in analyzer.numeric_cols if c != col_x], key="scatter_y")
            if st.button("Построить диаграмму рассеяния", key="scatter_btn"):
                show_scatter(df, col_x, col_y)
        else:
            show_info("Необходимо минимум два числовых признака.")

    # Корреляционная матрица
    with st.expander("🔴 Матрица корреляций", expanded=False):
        if len(analyzer.numeric_cols) >= 2:
            corr_cols = st.multiselect(
                "Выберите числовые признаки (мин. 2):",
                analyzer.numeric_cols,
                default=analyzer.numeric_cols[:min(4, len(analyzer.numeric_cols))]
            )
            if st.button("Построить матрицу корреляций", key="corr_btn"):
                show_correlation_heatmap(df, corr_cols)
        else:
            show_info("Недостаточно числовых признаков.")