"""Отображение метрик модели в виде st.metric."""
import streamlit as st
import pandas as pd

def show_classification_metrics(metrics_dict: dict) -> None:
    cols = st.columns(len(metrics_dict))
    for i, (name, value) in enumerate(metrics_dict.items()):
        with cols[i]:
            st.metric(label=name, value=f"{value:.3f}" if value is not None else "N/A")

def show_regression_metrics(metrics_dict: dict) -> None:
    cols = st.columns(len(metrics_dict))
    for i, (name, value) in enumerate(metrics_dict.items()):
        with cols[i]:
            st.metric(label=name, value=f"{value:.4f}" if value is not None else "N/A")

def show_clustering_metrics(metrics_dict: dict) -> None:
    cols = st.columns(len(metrics_dict))
    for i, (name, value) in enumerate(metrics_dict.items()):
        with cols[i]:
            st.metric(label=name, value=f"{value:.3f}" if value is not None else "N/A")