"""
Configuration file for dataset preparation.

Centralizes parameters (sampling rate, window size, thresholds) and
paths for input/output directories used by the Capture-24 pipeline.
"""

# -------------------------
# PARAMETERS
# -------------------------

SAMPLING_RATE = 100  # Hz
WINDOW_SECONDS = 5
WINDOW_SIZE_SAMPLES = int(SAMPLING_RATE * WINDOW_SECONDS)
MIN_SAMPLES = 250  # mínimo aceitável por janela
ANNOTATION_NA_THRESHOLD = 0.5  # descartar janela se >= 50% das annotation forem NA


LABEL_COLUMNS_TO_KEEP = [
    'label:Walmsley2020',
    'label:WillettsSpecific2018',
    'label:WillettsMET2018'
    ]

SEX_MAP = {'F': 1, 'M': 0}
AGE_MAP = {'18-29': 0, '30-37': 1, '38-52': 2, '53+': 3}

# ---------------------------------------------------------
# Label dictionaries
# ---------------------------------------------------------
WILLETS_LABELS = {
    0: 'bicycling',
    1: 'household-chores',
    2: 'manual-work',
    3: 'mixed-activity',
    4: 'sitting',
    5: 'sleep',
    6: 'sports',
    7: 'standing',
    8: 'vehicle',
    9: 'walking'
}

WALMSLEY_LABELS = {
    0: 'light',
    1: 'moderate-vigorous',
    2: 'sedentary',
    3: 'sleep'
}

# -------------------------
# MODEL PATHS
# -------------------------
#MODEL_4_PATH = "api/model/mlp_baseline_4classes.keras"
MLP_MODEL_10_PATH = "app/model/mlp_baseline_10classes.keras"
MLP_PREPROCESSOR_PATH = "app/model/preprocessor.joblib"
MLP_FEATURE_NAMES_PATH = "app/model/feature_names.joblib"

MODEL_RF_PATH = "app/model/randomforest.joblib"
RF_FEATURE_NAMES = ['x_mean', 'x_std', 'x_min', 'x_max',
                    'y_mean', 'y_std', 'y_min', 'y_max',
                    'z_mean', 'z_std', 'z_min', 'z_max',
                    'energy_x', 'energy_y', 'energy_z', 'energy_total',
                    'corr_xy', 'corr_xz', 'corr_yz',
                    'fft_dom_freq', 'fft_peak_power']
