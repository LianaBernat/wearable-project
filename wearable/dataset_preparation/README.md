# 📦 Dataset Preparation Module (`dataset_preparation/`)

This module contains the full pipeline responsible for transforming the raw **Capture-24** dataset into a clean, feature-rich, windowed dataset ready for machine learning modeling.

It performs the following steps:

1. 5-second time-based windowing
2. Numerical feature extraction from accelerometer signals
3. Annotation mapping and label simplification
4. Label encoding (fixed label scheme)
5. Chunk-based incremental Parquet generation
6. Merging chunks per participant
7. Final concatenation of all participants


## 📁 Module Structure

### `config.py`

Centralizes configuration parameters:

- Sampling rate and window duration
- Minimum samples per window
- Annotation completeness threshold
- FFT-related parameters
- Paths for raw and processed data
- Fixed demographic maps (`SEX_MAP`, `AGE_MAP`)


### `features.py`

Computes numerical features for each 5-second window.

Includes:

- Axis statistics (mean, standard deviation, min, max)
- Energy features (per axis and total)
- Magnitude-based features
- Axis correlations (`corr_xy`, `corr_xz`, `corr_yz`)
- Frequency-domain features using Welch’s method (`fft_dom_freq`, `fft_peak_power`)

Main functions:

 - `compute_window_features_chunked(window_df)`



### `annotation.py`

Handles annotation logic for Capture-24.

Responsibilities:

- Build mapping dictionaries (raw annotation → simplified labels)
- Apply majority vote within each 5-second window
- Detect ambiguous windows (ties → `"ambiguous"` with encoding `-1`)
- Encode simplified labels as integer IDs

Main functions:

- `build_annotation_maps(annotation_map_df, label_columns=LABEL_COLUMNS_TO_KEEP)`
- `map_annotations_and_encode(window_df, mapping_dicts, enc_maps, label_columns=LABEL_COLUMNS_TO_KEEP)`



### `participants_pipeline.py`

Core processing pipeline for individual participants.

Responsibilities:

- Chunked reading of large CSV/GZIP files
- Construction of 5-second, non-overlapping windows
- Window filtering (minimum samples, annotation completeness)
- Numerical feature extraction (via `features.py`)
- Label mapping + encoding (inline or via `annotation.py`)
- Adding metadata (participant ID, sex, age group)
- Adding temporal context (cyclical hour-of-day features)
- Saving incremental `chunk_XXX.parquet` files per participant

Main functions:

- `process_participant_file_chunked(...)`
- `process_all_participants(...)`

Typical output structure:

    participants/P001/chunk_000.parquet
    participants/P001/chunk_001.parquet
    ...
    participants/P002/chunk_000.parquet
    ...


### `join_chunks_per_participant.py`

Merges all chunk files for each participant into a single file:

    participants/P001/P001_full.parquet
    participants/P002/P002_full.parquet
    ...


### `join_participants.py`

Concatenates all participant-level datasets into a single final dataset:

    participants/Participants_all.parquet

This is the dataset consumed by the modeling pipeline.


## 🚀 Full Pipeline Overview

The typical end-to-end flow for dataset preparation is:

1. `process_all_participants()` (from `participants_pipeline.py`)
2. `join_chunks()` (from `join_chunks_per_participant.py`)
3. `join_participants()` (from `join_participants.py`)


## 📌 Notes

- The pipeline is designed for **large-scale** datasets and uses chunk-based reading to minimize RAM usage.
- The dataset preparation module is fully decoupled from the modeling code, which should live in a separate `modeling/` package.
