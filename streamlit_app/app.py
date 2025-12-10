
# streamlit_app/app.py
"""
Streamlit app to upload accelerometer data and display predicted activities
"""



# -----------------------------
# IMPORTS
# -----------------------------

from io import BytesIO
import math
import requests
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path


# -----------------------------
# PATHS / CONSTANTS / GUIDELINES
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
API_URL = "https://wearable-api-1009461955584.us-central1.run.app/predict"

WINDOW_SECONDS = 5          # janela do modelo (5s)
MAX_API_BYTES = 30 * 1024 * 1024  # 30MB

REQUIRED_COLUMNS = ["time", "x", "y", "z"]


GUIDELINES = {
    "18-29": {"sleep_min": 7, "sleep_max": 9, "mvpa_min": 0.5, "mvpa_max": 1.5, "sedentary_max": 8},
    "30-37": {"sleep_min": 7, "sleep_max": 9, "mvpa_min": 0.5, "mvpa_max": 1.5, "sedentary_max": 8},
    "38-52": {"sleep_min": 7, "sleep_max": 9, "mvpa_min": 0.5, "mvpa_max": 1.5, "sedentary_max": 8},
    "53+":   {"sleep_min": 7, "sleep_max": 8, "mvpa_min": 0.5, "mvpa_max": 1.5, "sedentary_max": 8},
}

SEDENTARY_10 = {"sitting", "vehicle"}
MVPA_10 = {"walking", "bicycling", "sports", "manual-work"}
SLEEP_10 = {"sleep"}



# -----------------------------
# FUNCTIONS
# -----------------------------

# LOAD PREDICTIONS FROM API

def call_api_with_parquet_bytes(parquet_bytes: bytes, original_name: str) -> dict:
    """
    Chama a API para UM parquet (bytes) e retorna o JSON.
    """
    files = {
        "file": (original_name, parquet_bytes, "application/octet-stream")
    }

    resp = requests.post(API_URL, files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()


def load_predictions(file_bytes: bytes, original_name: str, df_uploaded: pd.DataFrame) -> dict:
    """
    Se o arquivo for <=30MB, manda uma vez só.
    Se for >30MB, divide df_uploaded em N pedaços e manda N parquet menores.
    Junta as predições de todos os pedaços.
    Retorna:
      {
        "walmsley_4classes": DataFrame,
        "willetts_10classes": DataFrame
      }
    """

    total_size = len(file_bytes)

    if total_size <= MAX_API_BYTES:
        # 1 Request
        data = call_api_with_parquet_bytes(file_bytes, original_name)
        dfs = {}
        for key in ["willetts_10classes", "walmsley_4classes"]:
            df = pd.DataFrame.from_dict(data[key])
            df = df.rename(columns={"window_start": "timestamp", "label": "predicted_activity"})
            if "label_id" in df.columns:
                df = df.drop(columns=["label_id"])
            dfs[key] = df
        return dfs

    # MULTIPLE Requests
    n_chunks = math.ceil(total_size / MAX_API_BYTES)
    n_rows = len(df_uploaded)
    rows_per_chunk = math.ceil(n_rows / n_chunks)

    all_4 = []
    all_10 = []

    for i in range(n_chunks):
        start = i * rows_per_chunk
        end = min((i + 1) * rows_per_chunk, n_rows)
        if start >= end:
            continue

        df_chunk = df_uploaded.iloc[start:end]

        buffer = BytesIO()
        df_chunk.to_parquet(buffer, index=False)
        chunk_bytes = buffer.getvalue()

        try:
            data_chunk = call_api_with_parquet_bytes(chunk_bytes, f"{original_name}_part{i+1}.parquet")
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling prediction API for chunk {i+1}/{n_chunks}: {e}")
            st.stop()

        df_10_chunk = pd.DataFrame.from_dict(data_chunk["willetts_10classes"])
        df_4_chunk = pd.DataFrame.from_dict(data_chunk["walmsley_4classes"])

        for df, lst in [(df_10_chunk, all_10), (df_4_chunk, all_4)]:
            df = df.rename(columns={"window_start": "timestamp", "label": "predicted_activity"})
            if "label_id" in df.columns:
                df = df.drop(columns=["label_id"])
            lst.append(df)

    df_10_full = pd.concat(all_10, ignore_index=True)
    df_4_full = pd.concat(all_4, ignore_index=True)

    return {
        "willetts_10classes": df_10_full,
        "walmsley_4classes": df_4_full,
    }


# AGGREGATION FOR 4-CLASS FUNCTION
def aggregate_4class(dist_4: pd.DataFrame) -> dict:
    """
    dist_4 has columns: activity, count, percentage, hours
    We normalize activities to lowercase to avoid case mismatches.
    """
    # build dict with lowercase keys
    hours_by_act = {str(a).lower(): float(h) for a, h in zip(dist_4["activity"], dist_4["hours"])}

    sleep_h = hours_by_act.get("sleep", 0.0)
    sed_h = hours_by_act.get("sedentary", 0.0)
    mvpa_h = hours_by_act.get("moderate-vigorous", 0.0)
    light_h = hours_by_act.get("light", 0.0)

    return {
        "sleep_h": sleep_h,
        "sedentary_h": sed_h,
        "mvpa_h": mvpa_h,
        "light_h": light_h,
    }


# AGGREGATION FOR 10-CLASS FUNCTION
def aggregate_10class(dist_10: pd.DataFrame) -> dict:
    """
    dist_10 has columns: activity, count, percentage, hours
    Aggregate into sleep / sedentary / mvpa buckets.
    """
    hours_by_act = dict(zip(dist_10["activity"], dist_10["hours"]))

    sleep_h = sum(hours_by_act.get(a, 0.0) for a in SLEEP_10)
    sed_h = sum(hours_by_act.get(a, 0.0) for a in SEDENTARY_10)
    mvpa_h = sum(hours_by_act.get(a, 0.0) for a in MVPA_10)

    return {
        "sleep_h": float(sleep_h),
        "sedentary_h": float(sed_h),
        "mvpa_h": float(mvpa_h),
    }


# CLASSIFY AGAINST GUIDELINE RANGE
def classify_against_range(value: float, min_val: float | None, max_val: float | None) -> str:
    """
    Simple text classification vs guideline range.
    """
    if min_val is not None and value < min_val:
        return "below"
    if max_val is not None and value > max_val:
        return "above"
    return "within"


# AGGREGATE TO MINUTES FUNCTION
def aggregate_to_minutes(df: pd.DataFrame, label_col: str = "predicted_activity") -> pd.DataFrame:
    """
    Agrupa janelas (ex: 5s) por minuto, escolhendo o rótulo mais frequente.
    Retorna colunas:
      - minute (datetime floored to minute)
      - predicted_activity
      - n_windows (quantas janelas naquele minuto)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["minute"] = df["timestamp"].dt.floor("min")

    def most_frequent(x):
        return x.value_counts().idxmax()

    grouped = (
        df.groupby("minute")[label_col]
        .agg(most_frequent)
        .reset_index()
        .rename(columns={label_col: "predicted_activity"})
    )

    # quantas janelas por minuto (pode ser útil)
    counts = (
        df.groupby("minute")[label_col]
        .size()
        .rename("n_windows")
        .reset_index()
    )

    result = pd.merge(grouped, counts, on="minute", how="left")
    return result



# -----------------------------
# CONFIG AND UI LAYOUT
# -----------------------------

st.set_page_config(
    page_title="Wearable Activity Classifier", page_icon="⏱️", layout="wide"
)

st.title("Wearable Activity Classifier – Demo")
st.write(
    "Upload accelerometer data from a wearable device and preview the predicted activities "
    "using our machine learning model."
)

st.markdown("---")


# USER INPUT FORM
with st.form("user_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        user_name = st.text_input("Name", value="")
    with col2:
        age = st.selectbox("Age", options=["18-29", "30-37", "38-52", "53+"])
    with col3:
        sex = st.selectbox("Sex", options=["Male", "Female"])

    st.markdown("### Upload accelerometer data")
    uploaded_file = st.file_uploader(
        "Parquet file (accelerometer data)", type=["parquet"]
    )

    submitted = st.form_submit_button("Run analysis")

files = {"file": ("P043_no_annotations_small.parquet", uploaded_file)}
response = requests.post(API_URL, files=files)


#AFTER SUBISSION
if submitted:
    if not user_name:
        st.error("Please enter a name.")
        st.stop()

    if uploaded_file is None:
        st.error("Please upload a .parquet file.")
        st.stop()

    file_bytes = uploaded_file.getvalue()

    # Usa BytesIO para ler parquet sem “estragar” os bytes que vão pra API
    try:
        df_uploaded = pd.read_parquet(BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Error reading parquet file: {e}")
        st.stop()

    st.success("File uploaded successfully ✅")

    with st.expander("Preview of uploaded data"):
        st.dataframe(df_uploaded.head())

    # Chama API (com chunking se precisar)
    with st.spinner("Processing data and running the model..."):
        preds = load_predictions(file_bytes, uploaded_file.name, df_uploaded)
        df_4 = preds["walmsley_4classes"].copy()
        df_10 = preds["willetts_10classes"].copy()


    st.success("Model processed successfully ✅")

    df_4_min = aggregate_to_minutes(df_4)
    df_10_min = aggregate_to_minutes(df_10)

    # -----------------------------
    # REPORT
    # -----------------------------

    # ACTIVITY DISTRIBUTION
    st.markdown("---")
    st.header("Predicted Activity Distribution")

    # ---- 4-class distribution ----
    dist_4 = (
        df_4["predicted_activity"]
        .value_counts()
        .rename_axis("activity")
        .reset_index(name="count")
    )
    dist_4["duration_sec"] = dist_4["count"] * WINDOW_SECONDS
    total_sec_4 = dist_4["duration_sec"].sum()
    dist_4["hours"] = (dist_4["duration_sec"] / 3600).round(2)
    dist_4["percentage"] = (dist_4["duration_sec"] / total_sec_4 * 100).round(1)

    st.subheader("4-class model (walmsley_4classes)")
    st.dataframe(dist_4)
    st.bar_chart(dist_4.set_index("activity")["duration_sec"], use_container_width=True)

    # ---- 10-class distribution ----
    dist_10 = (
        df_10["predicted_activity"]
        .value_counts()
        .rename_axis("activity")
        .reset_index(name="count")
    )
    dist_10["duration_sec"] = dist_10["count"] * WINDOW_SECONDS
    total_sec_10 = dist_10["duration_sec"].sum()
    dist_10["hours"] = (dist_10["duration_sec"] / 3600).round(2)
    dist_10["percentage"] = (dist_10["duration_sec"] / total_sec_10 * 100).round(1)

    st.subheader("10-class model (willetts_10classes)")
    st.dataframe(dist_10)
    st.bar_chart(dist_10.set_index("activity")["duration_sec"], use_container_width=True)

    main_4 = dist_4.iloc[0]
    main_10 = dist_10.iloc[0]


    # TIMELINE
    st.markdown("---")
    st.header("Activity Timeline (Gantt-style)")

    # 4-class Gantt
    st.subheader("4-class model (walmsley_4classes)")
    if not df_4_min.empty:
        df_4_gantt = df_4_min.copy()
        df_4_gantt["start"] = df_4_gantt["minute"]
        df_4_gantt["end"] = df_4_gantt["minute"] + pd.Timedelta(minutes=1)
        df_4_gantt["track"] = "4-class"

        fig4 = px.timeline(
            df_4_gantt,
            x_start="start",
            x_end="end",
            y="track",
            color="predicted_activity",
            hover_data=["n_windows"],
        )
        fig4.update_yaxes(title=None, showticklabels=False)
        fig4.update_layout(showlegend=True, height=250)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No valid data for 4-class timeline.")

    # 10-class Gantt
    st.subheader("10-class model (willetts_10classes)")
    if not df_10_min.empty:
        df_10_gantt = df_10_min.copy()
        df_10_gantt["start"] = df_10_gantt["minute"]
        df_10_gantt["end"] = df_10_gantt["minute"] + pd.Timedelta(minutes=1)
        df_10_gantt["track"] = "10-class"

        fig10 = px.timeline(
            df_10_gantt,
            x_start="start",
            x_end="end",
            y="track",
            color="predicted_activity",
            hover_data=["n_windows"],
        )
        fig10.update_yaxes(title=None, showticklabels=False)
        fig10.update_layout(showlegend=True, height=250)
        st.plotly_chart(fig10, use_container_width=True)
    else:
        st.info("No valid data for 10-class timeline.")



    # PREDICTED WINDOWS
    st.markdown("---")
    st.header("Sample of Predicted Windows (Grouped by Minute)")


    df_4_min_ren = df_4_min[["minute", "predicted_activity"]].rename(
        columns={"predicted_activity": "activity_4classes"}
    )
    df_10_min_ren = df_10_min[["minute", "predicted_activity"]].rename(
        columns={"predicted_activity": "activity_10classes"}
    )

    merged_minute = (
        pd.merge(df_4_min_ren, df_10_min_ren, on="minute", how="outer")
        .sort_values("minute")
    )

    st.dataframe(merged_minute.head(50))



    # SUMMARY BASED ON GUIDELINES
    st.markdown("---")
    st.header("Summary vs Health Guidelines")


    # Intro
    st.markdown(
        f"""
        - For **{user_name} ({age} years, {sex})**, we compared predictions from **both models**.
        - **4-class model:** most frequent activity = **{main_4['activity']}**
        (~**{main_4['hours']:.1f} hours**, {main_4['percentage']:.1f}% of predicted windows).
        - **10-class model:** most frequent activity = **{main_10['activity']}**
        (~**{main_10['hours']:.1f} hours**, {main_10['percentage']:.1f}% of predicted windows).
        """
    )

    age_group = age
    g = GUIDELINES[age_group]

    st.markdown(
        f"""
        **Age group:** '{age_group}'\n
        Recommended (per day):\n
        • Sleep: **{g['sleep_min']}–{g['sleep_max']} h**\n
        • Moderate–vigorous activity (MVPA): **≥ {g['mvpa_min']:.1f} h** (~{int(g['mvpa_min']*60)} min)\n
        • Sedentary time (sitting/very low movement): **≤ {g['sedentary_max']} h**
        """
    )

    # Use COPIES for aggregation
    agg4 = aggregate_4class(dist_4.copy())
    agg10 = aggregate_10class(dist_10.copy())

    sleep4, sed4, mvpa4 = agg4["sleep_h"], agg4["sedentary_h"], agg4["mvpa_h"]
    sleep10, sed10, mvpa10 = agg10["sleep_h"], agg10["sedentary_h"], agg10["mvpa_h"]

    sleep4_status = classify_against_range(sleep4, g["sleep_min"], g["sleep_max"])
    mvpa4_status  = classify_against_range(mvpa4, g["mvpa_min"], None)
    sed4_status   = classify_against_range(sed4, None, g["sedentary_max"])

    sleep10_status = classify_against_range(sleep10, g["sleep_min"], g["sleep_max"])
    mvpa10_status  = classify_against_range(mvpa10, g["mvpa_min"], None)
    sed10_status   = classify_against_range(sed10, None, g["sedentary_max"])

    def format_status(status: str) -> str:
        """
        Map 'below' / 'within' / 'above' to a colored, readable label.
        """
        if status == "within":
            return "🟢 within guideline"
        if status == "below":
            return "🟡 below guideline"
        if status == "above":
            return "🔴 above guideline"
        return status

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("4-class model (walmsley_4classes)")
        st.markdown(
            f"""
            - **Sleep:** ~**{sleep4:.1f} h** → {format_status(sleep4_status)}
            - **Moderate–vigorous:** ~**{mvpa4:.1f} h** → {format_status(mvpa4_status)}
            - **Sedentary:** ~**{sed4:.1f} h** → {format_status(sed4_status)}
            - **Light activity:** ~**{agg4['light_h']:.1f} h**
            """
        )

    with col_b:
        st.subheader("10-class model (willetts_10clases)")
        st.markdown(
            f"""
            - **Sleep (sleep):** ~**{sleep10:.1f} h** → {format_status(sleep10_status)}
            - **MVPA (walking/bicycling/sports/manual-work):** ~**{mvpa10:.1f} h** → {format_status(mvpa10_status)}
            - **Sedentary (sitting/vehicle):** ~**{sed10:.1f} h** → {format_status(sed10_status)}
            """
        )

    merged_for_agreement = merged_minute.dropna(subset=["activity_4classes", "activity_10classes"]).copy()
    if not merged_for_agreement.empty:
        agree_pct = (
            (merged_for_agreement["activity_4classes"] == merged_for_agreement["activity_10classes"])
            .mean() * 100
        ).round(1)
        st.markdown(
            f"- On overlapping windows, both models assign the **same label** in about **{agree_pct}%** of cases."
        )
