"""
Модуль отрисовки графиков Plotly.
Все графики принимают данные и параметры, не зависят от UI-логики.
"""
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder

def show_histogram(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        st.error(f"Столбец {col} не найден.")
        return
    fig = px.histogram(df, x=col, nbins=30, title=f"Гистограмма: {col}")
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, width='stretch')

def show_scatter(df: pd.DataFrame, col_x: str, col_y: str) -> None:
    fig = px.scatter(df, x=col_x, y=col_y,
                     title=f"Диаграмма рассеяния: {col_x} vs {col_y}",
                     opacity=0.6)
    st.plotly_chart(fig, width='stretch')

def show_correlation_heatmap(df: pd.DataFrame, cols: list) -> None:
    if len(cols) < 2:
        st.warning("Минимум два числовых признака.")
        return
    corr = df[cols].corr()
    fig = ff.create_annotated_heatmap(
        z=corr.round(3).values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        annotation_text=corr.round(2).values
    )
    fig.update_layout(title="Матрица корреляций", width=700, height=600)
    st.plotly_chart(fig, width='stretch')

def show_boxplot(df: pd.DataFrame, col: str) -> None:
    fig = px.box(df, y=col, title=f"Box plot: {col}")
    st.plotly_chart(fig, width='stretch')

def show_class_balance(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        st.error(f"Столбец {target} не найден.")
        return
    counts = df[target].value_counts().reset_index()
    counts.columns = [target, 'count']
    fig = px.bar(counts, x=target, y='count',
                 title=f"Распределение классов для {target}",
                 text='count')
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, width='stretch')

def show_confusion_matrix(y_true, y_pred) -> None:
    """
    Отображает матрицу ошибок:
    строки (ось Y) — предсказанные классы,
    столбцы (ось X) — истинные классы.
    Классы отображаются в порядке возрастания исходных меток (например, 0, 1, 2, ...).
    """

    # Приводим метки к последовательным целым числам (0,1,2,...)
    le = LabelEncoder()
    all_labels = np.concatenate([y_true, y_pred])
    le.fit(all_labels)
    y_true_enc = le.transform(y_true)
    y_pred_enc = le.transform(y_pred)

    # Стандартная матрица: строки - истина, столбцы - предсказание
    cm = confusion_matrix(y_true_enc, y_pred_enc)
    # Транспонируем: строки - предсказание, столбцы - истина
    cm = cm.T
    cm = cm[:, [1, 0]]   # меняем местами первую и вторую строки

    # Исходные метки классов в порядке возрастания
    class_labels = [str(cls) for cls in sorted(le.classes_)]
    n = len(class_labels)

    annotations = [[str(cm[i, j]) for j in range(n)] for i in range(n)]

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=list(range(n)),      # числовые индексы, не сортируются как строки
        y=list(range(n)),
        annotation_text=annotations,
        colorscale='Blues',
        showscale=True
    )
    # Подписи оси X (истинные классы) – 0, 1
    fig.update_xaxes(
        tickvals=list(range(n)),
        ticktext=class_labels[::-1],          # ['0', '1']
        title_text='Истинный класс'
    )
    # Подписи оси Y (предсказанные классы) – 1, 0 (верхняя строка – 1)
    fig.update_yaxes(
        tickvals=list(range(n)),
        ticktext=class_labels,    # ['1', '0']
        title_text='Предсказанный класс'
    )
    fig.update_layout(title='Матрица ошибок')
    st.plotly_chart(fig, width='stretch')

def show_prediction_plot(y_true, y_pred) -> None:
    """
    Строит график "Фактические vs Предсказанные значения" для регрессии.
    Добавляет линию y=x для визуальной оценки качества.
    """
    import plotly.express as px
    import streamlit as st
    import pandas as pd

    df_plot = pd.DataFrame({'Фактические': y_true, 'Предсказанные': y_pred})
    fig = px.scatter(
        df_plot,
        x='Фактические',
        y='Предсказанные',
        title='Фактические vs Предсказанные значения',
        opacity=0.6
    )
    # Линия идеального совпадения
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_shape(
        type='line',
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color='red', dash='dash')
    )
    fig.update_layout(xaxis_title='Фактические', yaxis_title='Предсказанные')
    st.plotly_chart(fig, width='stretch')

def show_cluster_plot(X, labels) -> None:
    """
    Визуализация результатов кластеризации.
    - Если признак один: scatter plot по индексу образца.
    - Если признака два: scatter plot с этими признаками.
    - Если три и более: применяет PCA до двух компонент и строит scatter plot.
    """
    from sklearn.decomposition import PCA
    import pandas as pd

    n_features = X.shape[1]

    if n_features == 1:
        df_plot = pd.DataFrame({
            'Индекс': range(len(X)),
            'Значение': X.iloc[:, 0].values,
            'Кластер': labels.astype(str)
        })
        fig = px.scatter(
            df_plot, x='Индекс', y='Значение', color='Кластер',
            title='Кластеры (один признак)'
        )
    elif n_features == 2:
        df_plot = pd.DataFrame({
            'Признак 1': X.iloc[:, 0].values,
            'Признак 2': X.iloc[:, 1].values,
            'Кластер': labels.astype(str)
        })
        fig = px.scatter(
            df_plot, x='Признак 1', y='Признак 2', color='Кластер',
            title='Кластеры (два признака)'
        )
    else:
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        df_plot = pd.DataFrame({
            'PC1': components[:, 0],
            'PC2': components[:, 1],
            'Кластер': labels.astype(str)
        })
        fig = px.scatter(
            df_plot, x='PC1', y='PC2', color='Кластер',
            title='Кластеры (PCA-проекция)'
        )
    st.plotly_chart(fig, width='stretch')