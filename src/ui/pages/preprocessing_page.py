"""
Страница предобработки данных.
Все операции вызываются через фабрику обработчиков и единый метод process().
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from ...preprocessing.preprocessing_factory import PreprocessingFactory
from ..components.notifications import show_success, show_error, show_info
from ..components.data_preview import show_head
from ..components.charts import show_boxplot
from ..utils.session import get_data, set_data
from ..utils.validators import check_data_loaded, check_numeric_columns, check_categorical_columns

def render() -> None:
    st.title("🧹 Предобработка данных")
    df = get_data()
    if not check_data_loaded(df):
        return

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

    # Пропуски
    with st.expander("💧 Пропущенные значения"):
        missing_method = st.selectbox(
            "Метод обработки пропусков:",
            ("Удалить строки с пропусками", "Удалить столбцы с пропусками",
             "Заполнить средним", "Заполнить медианой", "Заполнить модой")
        )
        threshold_pct = st.slider("Порог пропусков (%):", 0, 100, 0, 1,
                                  help="Удаление/заполнение применяется, если доля пропусков >= порога.")
        threshold = threshold_pct / 100.0

        if missing_method != "Удалить столбцы с пропусками":
            target_cols = st.multiselect("Выберите столбцы:", df.columns.tolist())
        else:
            target_cols = None

        if st.button("Применить обработку пропусков"):
            if df.isnull().sum().sum() == 0:
                show_info("Пропуски отсутствуют.")
            else:
                handler = PreprocessingFactory.get_handler("missing")
                method_map = {
                    "Удалить строки с пропусками": "delete_rows",
                    "Удалить столбцы с пропусками": "delete_columns",
                    "Заполнить средним": "fill_mean",
                    "Заполнить медианой": "fill_median",
                    "Заполнить модой": "fill_mode"
                }
                method = method_map[missing_method]
                df_new = handler.process(df, method=method, columns=target_cols, threshold=threshold)
                set_data(df_new)
                show_success("Обработка выполнена. Данные обновлены.")
                show_head(df_new)

    # Дубликаты
    with st.expander("♊ Дубликаты"):
        dup_subset = st.multiselect(
            "Столбцы для проверки дубликатов (если не выбрано — все):",
            df.columns.tolist(), key="dup_subset"
        )
        dup_keep = st.selectbox("Какую запись оставить:", ('first', 'last', False), key="dup_keep")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Показать дубликаты"):
                subset = dup_subset if dup_subset else None
                dup_mask = df.duplicated(subset=subset, keep=False)
                if dup_mask.any():
                    st.warning(f"Найдено дубликатов: {dup_mask.sum()} строк.")
                    st.dataframe(df[dup_mask].head(10), width='stretch')
                else:
                    show_info("Дубликаты не найдены.")
        with col2:
            if st.button("🗑️ Удалить дубликаты"):
                before = df.shape[0]
                handler = PreprocessingFactory.get_handler("duplicates")
                df_new = handler.process(df, subset=dup_subset if dup_subset else None, keep=dup_keep)
                after = df_new.shape[0]
                set_data(df_new)
                show_success(f"Удалено {before - after} дубликатов. Данные обновлены.")
                show_head(df_new)

    # Удаление столбцов
    with st.expander("🗑️ Удаление столбцов"):
        cols_to_delete = st.multiselect("Выберите столбцы для удаления:", df.columns.tolist())
        if st.button("Удалить выбранные столбцы"):
            if not cols_to_delete:
                show_error("Выберите хотя бы один столбец.")
            else:
                handler = PreprocessingFactory.get_handler("columns")
                df_new = handler.process(df, columns=cols_to_delete)
                set_data(df_new)
                show_success(f"Удалено столбцов: {len(cols_to_delete)}. Данные обновлены.")
                show_head(df_new)

    # Выбросы
    with st.expander("📈 Выбросы (Z-score)"):
        if not check_numeric_columns(df, numeric_cols):
            return
        col = st.selectbox("Числовой столбец:", numeric_cols, key="outlier_col")
        threshold_z = st.slider("Порог Z-score:", 2.0, 5.0, 3.0, 0.1, key="z_threshold")
        action = st.radio("Действие с выбросами:", ("Удаление", "Замена средним", "Замена медианой"), key="outlier_action")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📊 Показать график выбросов"):
                show_boxplot(df, col)
        with c2:
            if st.button("⚙️ Применить"):
                action_map = {"Удаление": "delete", "Замена средним": "mean", "Замена медианой": "median"}
                handler = PreprocessingFactory.get_handler("outliers")
                df_new, num_outliers = handler.process(df, column=col, threshold=threshold_z, action=action_map[action])
                set_data(df_new)
                if action == "Удаление":
                    show_success(f"Выбросы удалены. Удалено строк: {num_outliers}. Осталось строк: {df_new.shape[0]}.")
                else:
                    show_success(f"Выбросы заменены на {action.split()[-1].lower()}. Заменено значений: {num_outliers}.")
                show_head(df_new)

    # Кодирование категориальных признаков
    with st.expander("🏷️ Кодирование категориальных признаков"):
        if not check_categorical_columns(df, categorical_cols):
            return
        encoding = st.selectbox("Тип кодирования:", ("Label Encoding", "One-Hot Encoding"))
        cat_cols_to_encode = st.multiselect("Выберите категориальные столбцы:", categorical_cols)
        if st.button("Применить кодирование"):
            if not cat_cols_to_encode:
                show_error("Выберите хотя бы один столбец.")
            else:
                handler = PreprocessingFactory.get_handler("categorical")
                df_new = handler.process(df, encoding="label" if encoding == "Label Encoding" else "onehot",
                                        columns=cat_cols_to_encode)
                set_data(df_new)
                show_success("Кодирование применено. Данные обновлены.")
                show_head(df_new)

    # Масштабирование
    with st.expander("📐 Масштабирование"):
        if not check_numeric_columns(df, numeric_cols):
            return
        scaling_method = st.selectbox("Метод масштабирования:", ("StandardScaler", "MinMaxScaler"))
        scale_cols = st.multiselect("Выберите числовые столбцы:", numeric_cols)
        if st.button("Применить масштабирование"):
            if not scale_cols:
                show_error("Выберите хотя бы один столбец.")
            else:
                handler = PreprocessingFactory.get_handler("scaling")
                df_new = handler.process(df, method="standard" if scaling_method == "StandardScaler" else "minmax",
                                        columns=scale_cols)
                set_data(df_new)
                show_success("Масштабирование выполнено.")
                show_head(df_new)