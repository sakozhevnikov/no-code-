"""Отображение DataFrame в Streamlit."""
import streamlit as st
import pandas as pd

def show_head(df: pd.DataFrame, n: int = 5) -> None:
    st.dataframe(df.head(n), width='stretch')

def show_tail(df: pd.DataFrame, n: int = 5) -> None:
    st.dataframe(df.tail(n), width='stretch')

def show_full(df: pd.DataFrame) -> None:
    st.dataframe(df, width='stretch')