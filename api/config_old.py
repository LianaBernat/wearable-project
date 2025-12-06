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
# Label dictionaries for predictions
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
# INPUT PATHS
# -------------------------
RAW_CAPTURE24_GLOB = ".../data/data_raw/capture24/P*.csv.gz"
METADATA_PATH = ".../data/data_raw/capture24/metadata.csv"
ANNOTATION_DICT_PATH = ".../data/data_raw/capture24/annotation-label-dictionary.csv"

# -------------------------
# OUTPUT PATHS
# -------------------------
OUT_DIR = ".../data/data_processed/participants"
PARTICIPANT_GLOB = "../data/data_raw/capture24/P*.csv.gz"

# -------------------------
# MLP_MODEL PATHS
# -------------------------
MLP_MODEL_4_PATH = "api/MLP_model/mlp_baseline_4classes.keras"
MLP_MODEL_10_PATH = "api/MLP_model/mlp_baseline_10classes.keras"
MLP_PREPROCESSOR_PATH = "api/MLP_model/preprocessor.joblib"
MLP_FEATURE_NAMES_PATH = "api/MLP_model/feature_names.joblib"
