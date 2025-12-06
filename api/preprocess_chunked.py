# api/preprocess_parquet_chunked.py

"""
Efficient chunked preprocessing for large Capture-24 parquet files.

This version uses PyArrow Scanner to stream through a large parquet file
without loading the entire dataset into memory. It processes the data in
batches, creates 5-second windows inside each batch, computes features,
and returns arrays ready for model prediction.

Inputs:
- .parquet file with columns: time, x, y, z
- trained preprocessor (StandardScaler)
- feature_names (ordered list)

Outputs:
- X_raw: pd.DataFrame with raw features
- X_ready: np.ndarray with scaled features
- window_starts: list of timestamps
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from datetime import timedelta

from api.features import compute_window_features_chunked
from api.config_old import WINDOW_SECONDS, MIN_SAMPLES


def preprocess_parquet_chunked(
    fileobj,
    preprocessor,
    feature_names,
    batch_size=200_000
):
    """
    Chunked preprocessing for parquet files using PyArrow Scanner.

    Parameters
    ----------
    fileobj : file-like
        Uploaded file from FastAPI (UploadFile.file)

    preprocessor : sklearn transformer
        Scaler object used during training

    feature_names : list
        Columns in the order expected by the model

    batch_size : int
        Number of rows per Arrow batch (default 200k)

    Returns
    -------
    X_ready : np.ndarray
    window_starts : list[datetime]
    X_raw
    """

    # --------------------------------------------------------------------
    # Step 1 — Load parquet using PyArrow
    # --------------------------------------------------------------------
    parquet_data = pq.ParquetFile(fileobj)

    all_feats = []
    all_window_starts = []

    # --------------------------------------------------------------------
    # Step 2 — Iterate over parquet batches
    # --------------------------------------------------------------------
    for batch in parquet_data.iter_batches(batch_size=batch_size, columns=['time', 'x', 'y', 'z']):

        df = batch.to_pandas(types_mapper=lambda t: 'float32' if pa.types.is_floating(t) else None)

        # Sanitize
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time', 'x', 'y', 'z'])
        if len(df) < MIN_SAMPLES:
            continue

        # ----------------------------------------------------------------
        # Step 3 — Compute time in seconds (relative to each batch)
        # ----------------------------------------------------------------
        t0 = df['time'].iloc[0]
        df['t_sec'] = (df['time'] - t0).dt.total_seconds()

        t_end = df['t_sec'].iloc[-1]
        window_starts_sec = np.arange(0, t_end, WINDOW_SECONDS)

        # ----------------------------------------------------------------
        # Step 4 — Build 5s windows inside the batch
        # ----------------------------------------------------------------
        for ws in window_starts_sec:
            we = ws + WINDOW_SECONDS
            mask = (df['t_sec'] >= ws) & (df['t_sec'] < we)
            wdf = df.loc[mask]

            if len(wdf) < MIN_SAMPLES:
                continue

            feats = compute_window_features_chunked(wdf)
            all_feats.append(feats)

            ws_dt = t0 + timedelta(seconds=float(ws))
            all_window_starts.append(ws_dt)

    # --------------------------------------------------------------------
    # Step 5 — Convert features to DataFrame
    # --------------------------------------------------------------------
    if len(all_feats) == 0:
        raise ValueError("No valid windows found in parquet file.")

    X_raw = pd.DataFrame(all_feats)

    # Verified columns order
    X_aligned = X_raw[feature_names]

    # --------------------------------------------------------------------
    # Step 6 — Apply scaler
    # --------------------------------------------------------------------
    X_ready = preprocessor.transform(X_aligned)

    return X_ready, all_window_starts
