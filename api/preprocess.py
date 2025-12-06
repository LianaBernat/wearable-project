"""
Preprocessing pipeline for deployment.
Executes the steps to transform raw accelerometer CSV data into
standardized features ready for model prediction.
Functions:
- load_preprocessor: Load the scaler and feature names.
- preprocess_capture24: Main function to preprocess a raw DataFrame.
    Parameters:
    - raw_df: pd.DataFrame loaded from CSV with columns ['time', 'x', 'y', 'z'].
    - preprocessor: Loaded StandardScaler object.
    - feature_names: List of feature names in the order expected by the model.
    Returns:
    - X_ready: np.ndarray of transformed features.
    - window_starts: List of timestamps for the start of each window.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from joblib import load

from .config_old import WINDOW_SECONDS, MIN_SAMPLES
from .features import compute_window_features_chunked


# --------------------------------------------------------------------
# 1) Load preprocessor and feature_names
# --------------------------------------------------------------------
def load_preprocessor(preprocessor_path="preprocessor.joblib",
                      feature_names_path="feature_names.joblib"):
    """
    Load the preprocessor (StandardScaler) and the order of features.
    """
    preprocessor = load(preprocessor_path)
    feature_names = load(feature_names_path)
    return preprocessor, feature_names


# --------------------------------------------------------------------
# 2) Main preprocess function
# --------------------------------------------------------------------
def preprocess_capture24(
    raw_df: pd.DataFrame,
    preprocessor,
    feature_names
):
    """
    Executes the entire preprocessing pipeline for deployment.

    Parameters
    ----------
    raw_df : pd.DataFrame
        DataFrame already loaded from the CSV containing columns:
            ['time', 'x', 'y', 'z']
        The 'annotation' column will be ignored if it exists.

    preprocessor : ColumnTransformer (StandardScaler)
        Object loaded via load_preprocessor().

    feature_names : list[str]
        Order of features expected by the model.

    Returns
    -------
    X_ready : np.ndarray
        Transformed features, ready for model.predict().

    window_starts : list[datetime]
        Timestamp of the start of each window (5s).

    """

    # ----------------------------------------------------------------
    # CSV Sanitization
    # ----------------------------------------------------------------
    # enforce minimum columns
    expected_cols = ['time', 'x', 'y', 'z']
    for col in expected_cols:
        if col not in raw_df.columns:
            raise ValueError(f"Missing required column: {col}")
    # keep only relevant columns
    df = raw_df[expected_cols].copy()

    # convert time
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time', 'x', 'y', 'z'])
    df = df.sort_values('time').reset_index(drop=True)

    if len(df) < MIN_SAMPLES:
        raise ValueError("File too small or invalid to generate windows.")

    # ----------------------------------------------------------------
    # Build relative time
    # ----------------------------------------------------------------
    t0 = df['time'].iloc[0]
    df['t_sec'] = (df['time'] - t0).dt.total_seconds()

    # ----------------------------------------------------------------
    # Create 5-second windows
    # ----------------------------------------------------------------
    t_end = df['t_sec'].iloc[-1]
    window_starts_sec = np.arange(0, t_end, WINDOW_SECONDS)

    rows = []
    window_starts = []

    for ws in window_starts_sec:
        we = ws + WINDOW_SECONDS
        mask = (df['t_sec'] >= ws) & (df['t_sec'] < we)
        wdf = df.loc[mask]

        # very small windows are discarded
        if len(wdf) < MIN_SAMPLES:
            continue

        # real timestamp of the window
        ws_datetime = t0 + timedelta(seconds=float(ws))

        # extract features
        feats = compute_window_features_chunked(wdf)

        rows.append(feats)
        window_starts.append(ws_datetime)

    # ----------------------------------------------------------------
    # Transform to DataFrame
    # ----------------------------------------------------------------
    X_raw_df = pd.DataFrame(rows)

    if len(X_raw_df) == 0:
        raise ValueError("No valid windows found. Check the CSV.")

    # ----------------------------------------------------------------
    # Ensure the order of features expected by the model
    # ----------------------------------------------------------------

    X_raw_aligned = X_raw_df[feature_names]

    # ----------------------------------------------------------------
    # Apply scaler
    # ----------------------------------------------------------------
    X_ready = preprocessor.transform(X_raw_aligned)

    return  X_raw_df.drop(columns="magnitude_mean"), X_ready, window_starts
