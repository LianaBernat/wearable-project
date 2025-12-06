# streamlit_app/app.py
"""
Streamlit app to upload accelerometer data and display predicted activities
"""


from pathlib import Path
import time

import json

import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# PATHS
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
API_DIR = BASE_DIR / "api"  # or "api" if your folder is lowercase
MODEL_DIR = API_DIR / "model"
JSON_PATH = BASE_DIR / "data" / "response_1765050557889.json"


# --- Guidelines for summary (hours per day) ---

GUIDELINES = {
    "18-29": {"sleep_min": 7, "sleep_max": 9, "mvpa_min": 0.5, "mvpa_max": 1.5, "sedentary_max": 8},
    "30-37": {"sleep_min": 7, "sleep_max": 9, "mvpa_min": 0.5, "mvpa_max": 1.5, "sedentary_max": 8},
    "38-52": {"sleep_min": 7, "sleep_max": 9, "mvpa_min": 0.5, "mvpa_max": 1.5, "sedentary_max": 8},
    "53+":   {"sleep_min": 7, "sleep_max": 8, "mvpa_min": 0.5, "mvpa_max": 1.5, "sedentary_max": 8},
}

SEDENTARY_10 = {"sitting", "vehicle"}
MVPA_10 = {"walking", "bicycling", "sports", "manual-work"}
SLEEP_10 = {"sleep"}



def classify_against_range(value: float, min_val: float | None, max_val: float | None) -> str:
    """
    Simple text classification vs guideline range.
    """
    if min_val is not None and value < min_val:
        return "below"
    if max_val is not None and value > max_val:
        return "above"
    return "within"


# Optional: if you'll call an API instead of loading a local model
# import requests

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Wearable Activity Classifier", page_icon="⏱️", layout="wide"
)

REQUIRED_COLUMNS = ["time", "x", "y", "z"]
WINDOW_SECONDS = 30

# -----------------------------
# Loading fake predictions for demo
# -----------------------------


def load_fake_predictions() -> dict:
    """
    Load fake predictions for both models from JSON.

    Expects JSON like:
    {
      "willetts_10classes": [...],
      "walmsley_4classes": [...]
    }

    Returns dict with two DataFrames:
      - preds["walmsley_4classes"]
      - preds["willetts_10classes"]
    """
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    dfs = {}

    for key in ["walmsley_4classes", "willetts_10classes"]:
        if key not in data:
            raise ValueError(
                f"Key '{key}' not found in JSON. Available: {list(data.keys())}"
            )

        records = data[key]
        df_model = pd.DataFrame(records)

        if "window_start" not in df_model.columns or "label" not in df_model.columns:
            raise ValueError(
                f"JSON list '{key}' must contain 'window_start' and 'label'."
            )

        df_model = df_model.rename(
            columns={"window_start": "timestamp", "label": "predicted_activity"}
        )

        if "label_id" in df_model.columns:
            df_model = df_model.drop(columns=["label_id"])

        dfs[key] = df_model

    return dfs

# -----------------------------
# Aggregation functions
# -----------------------------

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



# -----------------------------
# UI LAYOUT
# -----------------------------

st.title("Wearable Activity Classifier – Demo")
st.write(
    "Upload accelerometer data from a wearable device and preview the predicted activities "
    "using our machine learning model."
)

st.markdown("---")

# ----- USER INFO FORM -----
with st.form("user_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        user_name = st.text_input("Name", value="")
    with col2:
        age = st.selectbox("Age", options=["18-29", "30-37", "38-52", "53+"])
    with col3:
        sex = st.selectbox("Sex", options=["Male", "Female"])

    # Mapear label -> chave no JSON
    model_choice_map = {
        "4-class model (walmsley_4classes)": "walmsley_4classes",
        "10-class model (willetts_10classes)": "willetts_10classes",
    }

    st.markdown("### Upload accelerometer data")
    uploaded_file = st.file_uploader(
        "CSV or Parquet file (accelerometer data)", type=["csv", "parquet"]
    )

    submitted = st.form_submit_button("Run analysis")


# -----------------------------
# WHEN USER SUBMITS
# -----------------------------

if submitted:
    if not user_name:
        st.error("Please enter a name.")
        st.stop()

    if uploaded_file is None:
        st.error("Please upload a file (.csv or .parquet).")
        st.stop()

    # Ler o arquivo só para "validar" (não será usado na predição)
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            df_uploaded = pd.read_csv(uploaded_file)
        elif filename.endswith(".parquet"):
            df_uploaded = pd.read_parquet(uploaded_file)
        else:
            st.error("Invalid file type. Upload a .csv or .parquet file.")
            st.stop()
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
        OSError,
        UnicodeDecodeError,
    ) as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success("File uploaded successfully ✅")

    with st.expander("Preview of uploaded data"):
        st.dataframe(df_uploaded.head())

    # Simular processamento
    with st.spinner("Processing data and running the model..."):
        time.sleep(4)  # fake delay

        # Carregar predições do JSON em vez de rodar o modelo
        preds = load_fake_predictions()
        df_4 = preds["walmsley_4classes"].copy()
        df_10 = preds["willetts_10classes"].copy()

    st.success("Model processed successfully ✅")

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
    total_4 = dist_4["count"].sum()
    dist_4["percentage"] = (dist_4["count"] / total_4 * 100).round(1)
    dist_4["hours"] = (dist_4["count"] * WINDOW_SECONDS / 3600).round(2)

    st.subheader("4-class model (walmsley_4classes)")
    st.dataframe(dist_4)
    st.bar_chart(dist_4.set_index("activity")["count"], use_container_width=True)

    # ---- 10-class distribution ----
    dist_10 = (
        df_10["predicted_activity"]
        .value_counts()
        .rename_axis("activity")
        .reset_index(name="count")
    )
    total_10 = dist_10["count"].sum()
    dist_10["percentage"] = (dist_10["count"] / total_10 * 100).round(1)
    dist_10["hours"] = (dist_10["count"] * WINDOW_SECONDS / 3600).round(2)

    st.subheader("10-class model (willets_10clases)")
    st.dataframe(dist_10)
    st.bar_chart(dist_10.set_index("activity")["count"], use_container_width=True)


    main_4 = dist_4.iloc[0]
    main_10 = dist_10.iloc[0]


    # -----------------------------

    # TIMELINE

    st.markdown("---")
    st.header("Predicted Activity Over Time")


    # Ensure timestamps
    df_4["timestamp"] = pd.to_datetime(df_4["timestamp"], errors="coerce")
    df_10["timestamp"] = pd.to_datetime(df_10["timestamp"], errors="coerce")

    # ---- 4-class timeline ----
    st.subheader("Timeline – 4-class model")

    df_4_sorted = df_4.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
    if not df_4_sorted.empty:
        df_4_sorted["activity_code"] = df_4_sorted["predicted_activity"].astype("category").cat.codes
        sample_4 = df_4_sorted.iloc[:: max(1, len(df_4_sorted) // 500)]
        st.line_chart(sample_4.set_index("timestamp")[["activity_code"]], use_container_width=True)
        st.caption("Encoded activity classes over time (4-class model).")
    else:
        st.info("No valid timestamps for 4-class model.")

    # ---- 10-class timeline ----
    st.subheader("Timeline – 10-class model")

    df_10_sorted = df_10.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
    if not df_10_sorted.empty:
        df_10_sorted["activity_code"] = df_10_sorted["predicted_activity"].astype("category").cat.codes
        sample_10 = df_10_sorted.iloc[:: max(1, len(df_10_sorted) // 500)]
        st.line_chart(sample_10.set_index("timestamp")[["activity_code"]], use_container_width=True)
        st.caption("Encoded activity classes over time (10-class model).")
    else:
        st.info("No valid timestamps for 10-class model.")


    # -----------------------------

    # PREDICTED WINDOWS

    st.markdown("---")
    st.header("Sample of Predicted Windows (Both Models)")

    df_4_sample = df_4[["timestamp", "predicted_activity"]].rename(
        columns={"predicted_activity": "activity_4classes"}
    )
    df_10_sample = df_10[["timestamp", "predicted_activity"]].rename(
        columns={"predicted_activity": "activity_10classes"}
    )

    merged = (
        pd.merge(df_4_sample, df_10_sample, on="timestamp", how="outer")
        .sort_values("timestamp")
    )

    st.dataframe(merged.head(50))


    # -----------------------------

    # --- SUMMARY BASED ON GUIDELINES ---
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
        **Age group:** `{age_group}`
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
        st.subheader("10-class model (willets_10clases)")
        st.markdown(
            f"""
            - **Sleep (sleep):** ~**{sleep10:.1f} h** → {format_status(sleep10_status)}
            - **MVPA (walking/bicycling/sports/manual-work):** ~**{mvpa10:.1f} h** → {format_status(mvpa10_status)}
            - **Sedentary (sitting/vehicle):** ~**{sed10:.1f} h** → {format_status(sed10_status)}
            """
        )

    merged_for_agreement = merged.dropna(subset=["activity_4classes", "activity_10classes"]).copy()
    if not merged_for_agreement.empty:
        agree_pct = (
            (merged_for_agreement["activity_4classes"] == merged_for_agreement["activity_10classes"])
            .mean() * 100
        ).round(1)
        st.markdown(
            f"- On overlapping windows, both models assign the **same label** in about **{agree_pct}%** of cases."
        )
