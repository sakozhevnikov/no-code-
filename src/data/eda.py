"""
Модуль разведочного анализа данных (EDA).
Предоставляет класс для отображения основной информации и построения
графиков по запросу пользователя. Содержит кэшируемые методы для ускорения.
"""
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from typing import List, Tuple
from src.utils import compute_correlation, build_heatmap

# --- Вспомогательные кэшируемые функции ---

@st.cache_data
def get_numeric_cols_cached(df: pd.DataFrame) -> List[str]:
    """Возвращает список числовых колонок (кэшируемо)."""
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

@st.cache_data
def get_categorical_cols_cached(df: pd.DataFrame) -> List[str]:
    """Возвращает список категориальных колонок (кэшируемо)."""
    return [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

@st.cache_data
def compute_categorical_describe(df: pd.DataFrame, cat_cols: Tuple[str, ...]) -> pd.DataFrame:
    """Вычисляет описательную статистику для категориальных колонок (кэшируемо)."""
    cat_df = df[list(cat_cols)].astype('category')
    desc = cat_df.describe().transpose()
    missing_counts = cat_df.isnull().sum()
    desc['missing'] = missing_counts
    return desc[['count', 'unique', 'top', 'freq', 'missing']]

@st.cache_data
def compute_correlation(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    """Вычисляет корреляционную матрицу для выбранных числовых колонок (кэшируемо)."""
    return df[list(cols)].corr()

@st.cache_data
def build_histogram(df: pd.DataFrame, col: str, nbins: int = 30):
    """Строит и возвращает объект Figure histogram (кэшируемо)."""
    fig = px.histogram(df, x=col, nbins=nbins, title=f"Гистограмма: {col}")
    fig.update_layout(bargap=0.1)
    return fig

@st.cache_data
def build_scatter(df: pd.DataFrame, col_x: str, col_y: str):
    """Строит и возвращает объект Figure scatter (кэшируемо)."""
    fig = px.scatter(df, x=col_x, y=col_y,
                     title=f"Диаграмма рассеяния: {col_x} vs {col_y}",
                     opacity=0.6)
    return fig


class EDAAnalyzer:
    """Класс для разведочного анализа DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Определяем типы один раз (кэшируемо через вспомогательные функции)
        self.numeric_cols = get_numeric_cols_cached(self.df)
        self.categorical_cols = get_categorical_cols_cached(self.df)

    def show_basic_info(self) -> None:
        """Отображает размерность, типы, пропуски и описательную статистику."""
        st.subheader("📋 Основная информация о данных")
        st.write(f"**Строк:** {self.df.shape[0]}, **Столбцов:** {self.df.shape[1]}")

        st.write("**Типы признаков:**")
        st.dataframe(self.df.dtypes.rename("Тип").to_frame())

        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.write("**Количество пропусков (по признакам):**")
            st.dataframe(missing.rename("Пропуски").to_frame())
        else:
            st.write("**Пропуски отсутствуют.**")

        st.write("**Описательная статистика (числовые признаки):**")
        if self.numeric_cols:
            # describe() не кэшируем, т.к. это быстро и меняется редко
            st.dataframe(self.df[self.numeric_cols].describe())
        else:
            st.write("Числовых признаков нет.")

        # Статистика по категориальным признакам (кэшируемо)
        if self.categorical_cols:
            st.write("**Описательная статистика категориальных признаков:**")
            desc = compute_categorical_describe(self.df, tuple(self.categorical_cols))
            st.dataframe(desc, width='stretch')
        else:
            st.info("Нет категориальных признаков для вывода статистики.")

    def show_histogram(self, col: str) -> None:
        """Строит гистограмму для выбранного числового признака."""
        if col not in self.df.columns:
            st.error(f"Столбец '{col}' не найден.")
            return
        fig = build_histogram(self.df, col)
        st.plotly_chart(fig, use_container_width=True)

    def show_scatter(self, col_x: str, col_y: str) -> None:
        """Строит диаграмму рассеяния для двух числовых признаков."""
        if col_x not in self.df.columns or col_y not in self.df.columns:
            st.error("Один из выбранных столбцов не найден.")
            return
        fig = build_scatter(self.df, col_x, col_y)
        st.plotly_chart(fig, use_container_width=True)

    def show_correlation_heatmap(self, cols: List[str]) -> None:
        if len(cols) < 2:
            st.warning("Выберите минимум два числовых признака.")
            return
        valid_cols = [c for c in cols if c in self.numeric_cols]
        if len(valid_cols) < 2:
            st.error("Недостаточно числовых признаков для матрицы корреляций.")
            return
        if len(valid_cols) > 20:
            st.warning("Отображаются первые 20 выбранных признаков для производительности.")
            valid_cols = valid_cols[:20]

        corr = compute_correlation(self.df, tuple(valid_cols))
        fig = build_heatmap(corr)
        st.plotly_chart(fig, use_container_width=True)

    def run_full_eda(self) -> None:
        """Показывает базовую информацию (без графиков)."""
        self.show_basic_info()