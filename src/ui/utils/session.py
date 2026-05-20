"""
Централизованная работа с st.session_state.
Вся остальная UI-логика должна использовать только эти функции.
"""
import streamlit as st
import pandas as pd

def set_data(df: pd.DataFrame) -> None:
    st.session_state['data'] = df

def get_data() -> pd.DataFrame:
    return st.session_state.get('data', None)

def clear_data() -> None:
    if 'data' in st.session_state:
        del st.session_state['data']

def set_model(model, name: str) -> None:
    st.session_state['trained_model'] = model
    st.session_state['model_name'] = name

def get_model():
    return st.session_state.get('trained_model', None), st.session_state.get('model_name', '')

def set_file_type(ftype: str) -> None:
    st.session_state['file_type'] = ftype

def get_file_type() -> str:
    return st.session_state.get('file_type', '')