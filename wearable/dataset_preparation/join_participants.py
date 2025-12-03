"""
join_participants.py — Merges all <pid>_full.parquet files into one final dataset.

This module reads each participant's full dataset file
(e.g., P001_full.parquet, P002_full.parquet) located in the participants/
directory and concatenates them into a unified final dataset.

It does NOT compute features or perform windowing or annotation; it only merges
per-participant final files.

Output example:
    participants/Participants_all.parquet
"""

import pandas as pd
from pathlib import Path

from .config import (
    METADATA_PATH,      # path to metadata.csv (to get PID list)
    OUT_DIR,            # base dir of participants folders
)


def join_participants(metadata = METADATA_PATH, all_participant_path = OUT_DIR):

    """
    Joins all <pid>_full.parquet files into a single dataset.

    Parameters
    ----------
    metadata_path : str
        Path to metadata.csv containing a 'pid' column.

    all_participant_path : str
        Base directory where participant folders (P001/, P002/, ...) are stored.


    Returns
    -------
    pathlib.Path
        Path to the final merged dataset.
    """

    metadata_df = pd.read_csv(metadata)
    pid = metadata_df['pid']
    dfs = []

    for id in pid:
        participant_file =  Path(all_participant_path) / id / f"{id}_full.parquet"

        df = pd.read_parquet(participant_file)
        dfs.append(df)
        print(f"[OK] {id} successfully joined")

    full_df=pd.concat(dfs, ignore_index=True)
    output_path= Path(all_participant_path)/f"Participants_all.parquet"
    full_df.to_parquet(output_path, index=False)
    return output_path
