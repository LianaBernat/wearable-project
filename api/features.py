"""
features.py — Computes numerical features for 5-second acceleration windows.

Contains:
- mean, std, min, max for each axis (x, y, z)
- energy features
- magnitude-based features
- axis correlations
- frequency domain features (Welch PSD)

All functions operate on a pandas DataFrame containing columns:
['x', 'y', 'z'].

Returns dictionaries with feature names and values.
"""

import numpy as np
from scipy import signal

from .config_old import (
    SAMPLING_RATE,
    WINDOW_SIZE_SAMPLES,
)

def _safe_corr(a, b):
    if len(a) < 2:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def compute_window_features_chunked(window_df):
    """
    Computes all numerical features for a 5-second window of accelerometer data.

    Parameters
    ----------
    window_df : pandas.DataFrame
        Must contain numeric columns ['x', 'y', 'z'] with at least MIN_SAMPLES rows.

    Returns
    -------
    dict
        Keys include:
        - Axis statistics: x_mean, x_std, x_min, x_max, ...
        - Energy: energy_x, energy_y, energy_z, energy_total
        - Magnitude: magnitude_mean
        - Correlations: corr_xy, corr_xz, corr_yz
        - Frequency features: fft_dom_freq, fft_peak_power
    """

    x = window_df['x'].to_numpy()
    y = window_df['y'].to_numpy()
    z = window_df['z'].to_numpy()

    feats = {}

    # -----------------------------
    # Estatísticas por eixo
    # -----------------------------
    for axis, arr in [('x', x), ('y', y), ('z', z)]:
        feats[f"{axis}_mean"] = np.mean(arr)
        feats[f"{axis}_std"]  = np.std(arr)
        feats[f"{axis}_min"]  = np.min(arr)
        feats[f"{axis}_max"]  = np.max(arr)

    # -----------------------------
    # Energia
    # -----------------------------
    feats['energy_x'] = np.mean(x**2)
    feats['energy_y'] = np.mean(y**2)
    feats['energy_z'] = np.mean(z**2)
    feats['energy_total'] = np.mean(x**2 + y**2 + z**2)

    # -----------------------------
    # Magnitude
    # -----------------------------
    mag = np.sqrt(x**2 + y**2 + z**2)
    feats['magnitude_mean'] = np.mean(mag)

    # -----------------------------
    # Correlações
    # -----------------------------

    feats['corr_xy'] = _safe_corr(x, y)
    feats['corr_xz'] = _safe_corr(x, z)
    feats['corr_yz'] = _safe_corr(y, z)

    # -----------------------------
    # FFT completa
    # -----------------------------
    mag_dt = signal.detrend(mag)

    freqs, psd = signal.welch(
        mag_dt,
        fs=SAMPLING_RATE,
        nperseg=256,
        nfft=WINDOW_SIZE_SAMPLES
    )
    if len(psd) == 0:
        feats['fft_dom_freq'] = 0
        feats['fft_peak_power'] = 0

    elif np.all(np.isnan(psd)):
        feats['fft_dom_freq'] = np.nan
        feats['fft_peak_power'] = np.nan


    elif feats['energy_total'] < 1.07:
        feats['fft_dom_freq'] = 0
        feats['fft_peak_power'] = 0

    else:
        valid = (freqs >= 0.3) & (freqs <= 12)
        freqs_valid = freqs[valid]
        psd_valid   = psd[valid]

        if len(psd_valid) == 0:
            feats['fft_dom_freq'] = 0
            feats['fft_peak_power'] = 0
        else:
            idx = np.argmax(psd_valid)
            feats['fft_dom_freq'] = freqs_valid[idx]
            feats['fft_peak_power'] = psd_valid[idx]

    return feats
