"""
Общие утилиты и функции, используемые разными модулями системы.
Сюда вынесены кэшируемые вычисления корреляции и построения heatmap.
"""
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
from typing import Tuple
from src.config import HEATMAP_WIDTH, HEATMAP_HEIGHT

@st.cache_data
def compute_correlation(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    """Вычисляет корреляционную матрицу для выбранных числовых колонок."""
    return df[list(cols)].corr()

@st.cache_data
def build_heatmap(corr_df, width: int = HEATMAP_WIDTH, height: int = HEATMAP_HEIGHT):
    fig = ff.create_annotated_heatmap(
        z=corr_df.round(3).values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        annotation_text=corr_df.round(2).values
    )
    fig.update_layout(title="Матрица корреляций", width=width, height=height)
    return fig