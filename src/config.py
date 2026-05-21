"""
Централизованные настройки приложения.
Значения, которые могут потребовать изменения при настройке системы, вынесены сюда.
"""

# --- Данные ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
SUPPORTED_FILE_TYPES = ["csv", "xlsx"]

# --- Гиперпараметры по умолчанию (для ручного ввода) ---
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 5
DEFAULT_N_CLUSTERS = 3
DEFAULT_EPS = 0.5
DEFAULT_MIN_SAMPLES = 5

# --- Optuna ---
OPTUNA_N_TRIALS = 10
OPTUNA_RANDOM_STATE = 42

# --- Визуализация ---
HISTOGRAM_NBINS = 30
HEATMAP_WIDTH = 700
HEATMAP_HEIGHT = 600