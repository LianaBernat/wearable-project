"""
participants_pipeline.py — Chunked processing pipeline for Capture-24 data.

This module handles:
- reading each participant file in chunks
- constructing 5-second windows
- filtering invalid windows (min samples, annotation completeness)
- computing numerical features for each window
- generating simplified labels via majority vote
- adding metadata and cyclical hour features
- saving results incrementally as Parquet chunks

It integrates:
- windowing logic (time slicing)
- numerical feature extraction (features.py)
- annotation mapping and encoding (annotation.py)
- dataset-level configuration (config.py)

This module produces one output directory per participant, containing
chunk_000.parquet, chunk_001.parquet, etc.
"""

import os
import shutil
import time
from glob import glob
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

# Imports do seu pacote
from .config import (
    WINDOW_SECONDS,
    MIN_SAMPLES,
    ANNOTATION_NA_THRESHOLD,
    SEX_MAP,
    AGE_MAP,
    OUT_DIR,
    PARTICIPANT_GLOB,
    LABEL_COLUMNS_TO_KEEP,
)

from .features import compute_window_features_chunked

def choose_chunksize():
    """Automatically selects chunk size based on available memory."""
    free_gb = psutil.virtual_memory().available / 1e9

    if free_gb >= 8:
        return 1_200_000
    elif free_gb >= 4:
        return 750_000
    else:
        return 500_000


def process_participant_file_chunked(
    path,
    metadata_df,
    mapping_dicts,
    enc_maps,
    label_columns,
    out_dir=OUT_DIR,
):
    """
    Processes ONE participant using chunks.
    Save incrementally (append) to Parquet.
    """
    filename = os.path.basename(path)     # "P001.csv.gz"
    pid = os.path.splitext(filename)[0]   # "P001.csv"
    pid = os.path.splitext(pid)[0]        # "P001"

    print(f"\n=== Processando {pid} ===")

    # Obter metadata
    meta = metadata_df.loc[metadata_df["pid"] == pid].iloc[0]
    sex_code = SEX_MAP.get(meta["sex"], np.nan)
    age_code = AGE_MAP.get(meta["age"], np.nan)

    # Caminho de saída

    # Diretório específico do participante
    out_dir_pid = os.path.join(out_dir, pid)
    if Path(out_dir_pid).exists():
        shutil.rmtree(out_dir_pid)
    os.makedirs(out_dir_pid, exist_ok=True)


    # Escolher chunksize pelo estado atual da memória
    chunksize = choose_chunksize()
    print(f"Usando chunksize: {chunksize:,}")

    reader = pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=['time', 'x', 'y', 'z', 'annotation'],
        dtype={'annotation': 'string'}
    )

    total_windows = 0
    total_valid = 0

    for chunk in tqdm(reader, desc=f"{pid} — chunks"):

        # Garantir tipos corretos
        chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce')
        chunk = chunk.dropna(subset=['time', 'x', 'y', 'z']).reset_index(drop=True)
        if len(chunk) == 0:
            continue

        # Tempo relativo dentro do chunk
        t0 = chunk['time'].iloc[0]
        chunk['t_sec'] = (chunk['time'] - t0).dt.total_seconds()

        # Construir janelas
        t_end = chunk['t_sec'].iloc[-1]
        window_starts = np.arange(0, t_end, WINDOW_SECONDS)

        rows_out = []

        for ws in window_starts:
            we = ws + WINDOW_SECONDS
            mask = (chunk['t_sec'] >= ws) & (chunk['t_sec'] < we)
            wdf = chunk.loc[mask]

            total_windows += 1

            if len(wdf) < MIN_SAMPLES:
                continue
            if wdf['annotation'].isna().mean() >= ANNOTATION_NA_THRESHOLD:
                continue

            total_valid += 1

            # Extrair features (PARTE 1)
            feats = compute_window_features_chunked(wdf)

            # Labels: mapeamento + majority vote
            ann = wdf["annotation"].astype(str)

            for col in label_columns:
                mapped = ann.map(mapping_dicts[col])
                counts = mapped.value_counts()
                top = counts.max()
                winners = counts[counts == top].index.tolist()

                if len(winners) > 1:
                    major = "ambiguous"
                    enc = -1
                else:
                    major = winners[0]
                    enc = enc_maps[col][major]

                feats[col] = major
                feats[col + "_enc"] = enc

            # Hora cíclica
            ws_datetime = t0 + timedelta(seconds=float(ws))
            frac_hour = (
                ws_datetime.hour
                + ws_datetime.minute / 60
                + ws_datetime.second / 3600
                + ws_datetime.microsecond / (3600 * 1e6)
            )
            frac_day = frac_hour / 24

            feats["hour_sin"] = np.sin(2 * np.pi * frac_day)
            feats["hour_cos"] = np.cos(2 * np.pi * frac_day)

            # Metadata
            feats["pid"] = pid
            feats["sex"] = sex_code
            feats["age_group"] = age_code

            # Auditoria
            feats["window_start"] = ws_datetime
            feats["window_end"] = ws_datetime + timedelta(seconds=WINDOW_SECONDS)
            feats["n_samples"] = len(wdf)
            feats["duration_seconds"] = (
                (wdf["time"].iloc[-1] - wdf["time"].iloc[0]).total_seconds()
                if len(wdf) >= 2 else 0
            )

            rows_out.append(feats)

        # Salvamento seguro por chunk (SEM append)
        if rows_out:
            df_out = pd.DataFrame(rows_out)

            # gerar nome incremental chunk_000.parquet, chunk_001.parquet...
            existing = [f for f in os.listdir(out_dir_pid) if f.endswith(".parquet")]
            chunk_id = len(existing)
            chunk_path = os.path.join(out_dir_pid, f"chunk_{chunk_id:03d}.parquet")

            #reorganizando as colunas
            audit_cols = ["pid", "window_start", "window_end","n_samples",
                          "duration_seconds", "sex", "age_group"]

            label_cols = ["label:Walmsley2020", "label:Walmsley2020_enc",
                          "label:WillettsSpecific2018", "label:WillettsSpecific2018_enc",
                          "label:WillettsMET2018", "label:WillettsMET2018_enc",]

            feature_cols = [c for c in df_out.columns
                if c not in audit_cols + label_cols]

            df_out = df_out[audit_cols + label_cols + feature_cols]

            df_out.to_parquet(chunk_path, index=False)

    return out_dir_pid


def process_all_participants(
    pattern=PARTICIPANT_GLOB,
    metadata_df=None,
    mapping_dicts=None,
    enc_maps=None,
    label_columns=LABEL_COLUMNS_TO_KEEP,
    out_dir=OUT_DIR
):
    """
    Processa TODOS os participantes usando o pipeline chunked.

    - pattern: glob com caminho dos arquivos PXXX.csv.gz
    - metadata_df: dataframe carregado previamente
    - mapping_dicts, enc_maps: gerados por build_annotation_maps
    - label_columns: lista das colunas simplificadas que serão usadas
    - out_dir: diretório base para saída

    Retorna lista dos diretórios gerados (1 por participante).
    """

    files = sorted(glob(pattern))
    if len(files) == 0:
        print("Nenhum arquivo encontrado com o padrão:", pattern)
        return []

    print(f"=== Iniciando processamento de {len(files)} participantes ===\n")

    dirs_out = []
    t0 = time.time()

    for f in files:
        try:
            out_pid = process_participant_file_chunked(
                path=f,
                metadata_df=metadata_df,
                mapping_dicts=mapping_dicts,
                enc_maps=enc_maps,
                label_columns=label_columns,
                out_dir=out_dir
            )
            dirs_out.append(out_pid)

        except Exception as e:
            print(f"\nERRO ao processar {f}:\n{e}\n")
            continue

    t1 = time.time()
    print(f"\n=== Finalizado processamento de todos os participantes ===")
    print(f"Tempo total: {(t1 - t0)/60:.2f} minutos\n")

    return dirs_out
