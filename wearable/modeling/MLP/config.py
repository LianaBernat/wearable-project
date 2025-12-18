"""
Configuration file for MLP model training.

Centralizes parameters related to feature selection, preprocessing,
model architecture and training for the Capture-24 MLP classifier.
"""
# -------------------------
# DATA & LABELS
# -------------------------

TARGET_COLUMN = "label:WillettsSpecific2018"
TARGET_ENC_COLUMN = "label:WillettsSpecific2018_enc"

N_CLASSES = 10
TEST_PID = "P043"

# -------------------------
# FEATURES
# -------------------------

FEATURES_ACC = [
    'x_mean', 'x_std','x_min', 'x_max',
    'y_mean', 'y_std', 'y_min', 'y_max',
    'z_mean','z_std', 'z_min', 'z_max',
    'energy_x', 'energy_y', 'energy_z','energy_total',
    'magnitude_mean', 'corr_xy', 'corr_xz', 'corr_yz',
    'fft_dom_freq', 'fft_peak_power'
]

FEATURES_CONTEXT = [
    'sex',
    'age_group',
    'hour_sin',
    'hour_cos'
]

# -------------------------
# PREPROCESSING
# -------------------------

CATEGORICAL_FEATURES = ["age_group"]
PASSTHROUGH_FEATURES = ["sex"]

FILLNA_ZERO_COLUMNS = ["corr_xy", "corr_xz", "corr_yz"]

# -------------------------
# MODEL
# -------------------------

MLP_HIDDEN_LAYERS = [256, 128, 64]
DROPOUT_RATES = [0.4, 0.3, 0.2]

LEARNING_RATE = 1e-3

# -------------------------
# TRAINING
# -------------------------

BATCH_SIZE = 128
EPOCHS = 100
VALIDATION_SPLIT = 0.2

EARLY_STOPPING_PATIENCE = 8

# -------------------------
# PATHS
# -------------------------

DATASET_PATH = ".../data/data_processed/participants/Participants_all.parquet"
MODEL_OUTPUT_DIR = ".../models/mlp"
