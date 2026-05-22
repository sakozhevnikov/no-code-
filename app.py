"""
Точка входа в приложение. Минималистичный: только конфигурация, боковое меню и вызов страниц.
"""
import streamlit as st
from src.ui.components.sidebar import render_sidebar
from src.ui.pages import upload_page, eda_page, preprocessing_page, ml_page

st.set_page_config(
    page_title="Интеллектуальный EDA",
    layout="wide",
    initial_sidebar_state="expanded"
)

page = render_sidebar()

if page == "📂 Загрузка данных":
    upload_page.render()
elif page == "📊 Предварительный анализ":
    eda_page.render()
elif page == "🧹 Предобработка данных":
    preprocessing_page.render()
elif page == "🤖 Машинное обучение":
    ml_page.render()