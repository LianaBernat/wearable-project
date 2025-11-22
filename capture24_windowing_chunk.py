import os
from glob import glob
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy import signal
from tqdm import tqdm
import psutil
import time


SAMPLING_RATE = 100  # Hz
WINDOW_SECONDS = 5
WINDOW_SIZE_SAMPLES = int(SAMPLING_RATE * WINDOW_SECONDS)  # 500
MIN_SAMPLES = 250  # mínimo aceitável por janela
ANNOTATION_NA_THRESHOLD = 0.5  # descartar janela se >= 50% das annotation forem NA
FFT_NFFT = WINDOW_SIZE_SAMPLES  # usar zero-padding até 500 quando necessário
LABEL_COLUMNS_TO_KEEP = ['label:Walmsley2020', 'label:WillettsSpecific2018', 'label:WillettsMET2018']


# Map para sexo e age
SEX_MAP = {'F': 1, 'M': 0}
AGE_MAP = {'18-29': 0, '30-37': 1, '38-52': 2, '53+': 3}

# Diretórios de entrada/saída (AJUSTAR ESSA PARTE)
PARTICIPANT_GLOB = "data/data_raw/capture24/capture24/P*.csv.gz"
METADATA_PATH = "data/data_raw/capture24/capture24/metadata.csv"
ANNOT_DICT_PATH = "data/data_raw/capture24/capture24/annotation-label-dictionary.csv"
OUT_DIR = "data/data_processed/participants"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Funções utilitárias ---------- #
def safe_parse_time(df, time_col='time'):
    """Assegura que coluna time seja datetime e ordena por time."""
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values(time_col).reset_index(drop=True)
    return df


def compute_window_features_chunked(window_df):
    """
    Extrai features de aceleração + FFT completa para uma janela.
    NÃO inclui metadata, labels ou hora cíclica.
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
    def safe_corr(a, b):
        if len(a) < 2:
            return np.nan
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    feats['corr_xy'] = safe_corr(x, y)
    feats['corr_xz'] = safe_corr(x, z)
    feats['corr_yz'] = safe_corr(y, z)

    # -----------------------------
    # FFT completa
    # -----------------------------
    mag_dt = signal.detrend(mag)

    freqs, psd = signal.welch(
        mag_dt,
        fs=100,
        nperseg=256,
        nfft=500
    )

    if np.all(np.isnan(psd)):
        feats['fft_dom_freq'] = np.nan
        feats['fft_peak_power'] = np.nan
    else:
        idx = np.argmax(psd)
        feats['fft_dom_freq'] = freqs[idx]
        feats['fft_peak_power'] = psd[idx]

    return feats


def build_annotation_maps(annotation_map_df, label_columns):
    """
    Cria dois dicionários globais:
      - mapping_dicts[col]: mapa {annotation_original → label_simplificado}
      - enc_maps[col]: mapa {label_simplificado → inteiro}

    Deve ser chamado UMA ÚNICA VEZ antes do processamento de janelas.
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

def map_annotations_and_encode(window_df, mapping_dicts, enc_maps, label_columns):
    """
    Faz o mapeamento da coluna 'annotation' original para os rótulos simplificados,
    realiza majority vote e aplica encoding fixo.

    - Assume que a janela já passou pelos filtros (>= 250 amostras e < 50% NA).
    - Se houver empate → 'ambiguous' com encoding -1.
    """

    ann = window_df['annotation'].astype(str)
    result = {}

    for col in label_columns:

        mapped = ann.map(mapping_dicts[col])

        # Majority vote (len(counts) nunca é 0 após filtros)
        counts = mapped.value_counts()
        top_count = counts.iloc[0]
        top_labels = counts[counts == top_count].index.tolist()

        if len(top_labels) > 1:
            major = "ambiguous"
        else:
            major = top_labels[0]

        result[col] = major

        # Encoding fixo (consistente para todas as janelas)
        if major == "ambiguous":
            result[col + "_enc"] = -1
        else:
            result[col + "_enc"] = enc_maps[col][major]

    return result


def choose_chunksize():
    """Escolhe chunksize automaticamente baseado na memória disponível."""
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
    Processa UM participante usando chunks.
    Salva incrementalmente (append) em parquet.
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

# ===========================================================
# Bloco principal (executa somente se rodado como script)
# ===========================================================
if __name__ == "__main__":

    print("\n=== Preparando pipeline CAPTURE-24 ===\n")

    # 1) Carregar metadata e annotation map
    metadata_df = pd.read_csv(METADATA_PATH)
    annotation_map_df = pd.read_csv(ANNOT_DICT_PATH)

    # 2) Construir dicts de mapeamento global
    mapping_dicts, enc_maps = build_annotation_maps(
        annotation_map_df,
        LABEL_COLUMNS_TO_KEEP
    )

    # 3) Rodar processamento completo
    output_dirs = process_all_participants(
        pattern=PARTICIPANT_GLOB,
        metadata_df=metadata_df,
        mapping_dicts=mapping_dicts,
        enc_maps=enc_maps,
        label_columns=LABEL_COLUMNS_TO_KEEP,
        out_dir=OUT_DIR
    )

    print("\n=== Pipeline concluído com sucesso! ===")
    print("Pastas produzidas:\n")
    for d in output_dirs:
        print(" -", d)
