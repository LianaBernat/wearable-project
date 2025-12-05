"""
Configuration file for Random Forest.

Centralizes parameters (sampling rate, window size, thresholds) and
paths for input/output directories used by the Capture-24 pipeline.
"""

# -------------------------
# Data Settings
# -------------------------

PATH = '../../data/data_processed/participants'
FILE = 'Participants_all.parquet'
DATAFILE = f"{PATH}/{FILE}"

# -------------------------
# Model Settings
# -------------------------
TARGET_LABELS = {
    1: "label:WillettsMET2018_enc",
    2: "label:WillettsSpecific2018_enc",
    3: "label:Walmsley2020_enc"
}

DEFAULT_LABEL_CHOICE = 2   # escolha automática da label sem input do usuário

TEST_SIZE = 0.3
RANDOM_STATE = 42

SMOTE_K = 3
SMOTE_STRATEGY = "not majority"

RF_N_ESTIMATORS = 150
RF_MAX_DEPTH = 10
RF_CLASS_WEIGHT = "balanced"
RF_JOBS = -1
