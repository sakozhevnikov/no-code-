"""
Анализатор несбалансированности классов.
Строит распределение классов для выбранного категориального признака.
"""
import streamlit as st
import plotly.express as px

class ImbalanceAnalyzer:
    """Визуализирует распределение категориального признака (целевой переменной)."""

    @staticmethod
    def show(df, categorical_cols: list):
        """Интерактивный выбор категориального столбца и отображение bar plot."""
        if not categorical_cols:
            st.info("Нет категориальных признаков для отображения распределения.")
            return
        col = st.selectbox("Выберите признак (целевую переменную):", categorical_cols)
        if st.button("Показать распределение классов", key="imbalance_btn"):
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = [col, 'count']
            fig = px.bar(value_counts, x=col, y='count',
                         title=f"Распределение классов для {col}",
                         text='count')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)