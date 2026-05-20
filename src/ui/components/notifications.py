"""Единый модуль уведомлений."""
import streamlit as st

def show_success(msg: str) -> None:
    st.success(msg)

def show_error(msg: str) -> None:
    st.error(msg)

def show_warning(msg: str) -> None:
    st.warning(msg)

def show_info(msg: str) -> None:
    st.info(msg)