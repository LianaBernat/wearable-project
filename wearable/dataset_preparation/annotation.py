"""
annotation.py — Handles annotation mapping, majority vote, and label encoding
for Capture-24 5-second windows.

This module provides:
- build_annotation_maps: creates dictionaries for mapping raw annotations
  (e.g., Walmsley2020 or Willetts2018 scheme) to simplified labels and
  corresponding integer encodings.

- map_annotations_and_encode: applies the mapping to a single window and
  performs majority vote, returning the **most frequent simplified label** that
  appears in the window

If two or more labels tie for the highest frequency, the window is marked
as "ambiguous", and the encoded value is set to -1.

These functions operate independently of numerical feature extraction and
are intended to be called by the windowing/pipeline module after computing
features for each 5-second window.
"""
import pandas as pd
from .config import LABEL_COLUMNS_TO_KEEP


def build_annotation_maps(annotation_map_df, label_columns=LABEL_COLUMNS_TO_KEEP):
    """
    Builds mapping dictionaries for raw annotations → simplified labels
    and simplified labels → integer encoding.

    Parameters
    ----------
    annotation_map_df : pandas.DataFrame
        Must contain a column 'annotation' (raw labels) and the simplified
        label columns defined in LABEL_COLUMNS_TO_KEEP.

    label_columns : list of str
        Columns containing the simplified annotation labels.

    Returns
    -------
    mapping_dicts : dict
        For each simplified label column:
            { raw_annotation_str → simplified_label_str }

    enc_maps : dict
        For each simplified label column:
            { simplified_label_str → integer_code }
    """

    mapping_dicts = {}
    enc_maps = {}

    for col in label_columns:
        # Mapeamento original → simplificado
        mapping = dict(zip(annotation_map_df['annotation'].astype(str),
                           annotation_map_df[col].astype(str)))
        mapping_dicts[col] = mapping

        # Encoding fixo baseado no conjunto completo de rótulos simplificados
        unique_labels = sorted(annotation_map_df[col].dropna().unique().tolist())
        enc_maps[col] = {lab: i for i, lab in enumerate(unique_labels)}

    return mapping_dicts, enc_maps


def map_annotations_and_encode(window_df, mapping_dicts, enc_maps, label_columns=LABEL_COLUMNS_TO_KEEP):
    """
    Maps raw annotation values to simplified labels and applies majority vote
    per window. Ambiguous windows (ties) receive encoding -1.

    The returned labels correspond to the **most frequent simplified label**
    that appears inside the 5-second window (majority vote). If two or more
    labels are tied for highest frequency, the window is marked as
    "ambiguous" and the encoded value is set to -1.

    Parameters
    ----------
    window_df : pandas.DataFrame
        Must contain a column "annotation" with the raw annotation strings.

    mapping_dicts : dict
        Output of build_annotation_maps.

    enc_maps : dict
        Output of build_annotation_maps.

    label_columns : list of str
        Simplified label columns to compute.

    Returns
    -------
    dict
        For each label column, returns:
            - the most frequent simplified label in the window
            - an integer encoding of that label as <column>_enc
    """

    ann = window_df['annotation'].astype(str)
    result = {}

    for col in label_columns:

        mapped = ann.map(mapping_dicts[col])

        # Majority vote: counts always has at least 1 element because windows with
        # >=50% NA or very small samples are filtered earlier in the pipeline.
        counts = mapped.value_counts()
        top_count = counts.iloc[0]
        top_labels = counts[counts == top_count].index.tolist()

        if len(top_labels) > 1:
            major = "ambiguous"
        else:
            major = top_labels[0]

        result[col] = major

        # fixed encoding
        if major == "ambiguous":
            result[col + "_enc"] = -1
        else:
            result[col + "_enc"] = enc_maps[col][major]

    return result
