# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from tensorflow import keras


# Optional: if you'll call an API instead of loading a local model
# import requests

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Wearable Activity Classifier",
    page_icon="⏱️",
    layout="wide"
)

REQUIRED_COLUMNS = ["timestamp", "x", "y", "z"]
MODEL_DIR = Path("model")



# -----------------------------
# MODEL / PIPELINE PLACEHOLDERS
# -----------------------------
@st.cache_resource
def load_artifacts():
    """
    Load preprocessor, feature names and Keras model once.
    Cached by Streamlit so it doesn't reload every time.
    """
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")

    # Optional: may be useful later for explanations
    try:
        feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
    except Exception:
        feature_names = None

    # For now we'll use the 4-classes model.
    # If you want the 10-classes one, change the filename here.
    model = keras.models.load_model(MODEL_DIR / "map_baseline_4classes.keras")

    return {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "model": model,
    }



def prepare_features(df: pd.DataFrame, preprocessor) -> np.ndarray:
    """
    Apply the SAME preprocessing as you used during training.

    Assumption:
    - `preprocessor` is a scikit-learn transformer (ColumnTransformer, Pipeline, etc.)
      saved with joblib.
    - It expects the raw dataframe with the same columns as in training.

    If during training you selected only some columns, reflect that here.
    """
    # Example: keep just the raw sensor columns if that’s what you used
    # Adjust according to your training code
    # X_raw = df[["acc_x", "acc_y", "acc_z"]].copy()

    X_raw = df.copy()  # more permissive: pass full df to preprocessor

    X = preprocessor.transform(X_raw)
    return X



def predict_activities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the model and return predictions merged with the original dataframe.
    """

    artifacts = load_artifacts()
    preprocessor = artifacts["preprocessor"]
    model = artifacts["model"]

    # 1) Prepare features with the saved preprocessor
    X = prepare_features(df, preprocessor)

    # 2) Predict with Keras model
    # Most likely you trained a softmax classifier, so `predict` returns probabilities.
    probas = model.predict(X, verbose=0)
    y_idx = probas.argmax(axis=1)

    # TODO: Plug label mapping if you have a label_encoder saved somewhere.
    # For now we’ll just show the class indices.
    # If you know the order of classes, define them here and map:
    #
    # class_labels = ["Sleeping", "Running", "Sitting", "Walking"]
    # y_labels = [class_labels[i] for i in y_idx]
    #
    # For now, just use numeric indices:
    y_labels = y_idx

    results = df.copy()
    results["predicted_activity"] = y_labels

    return results


# If you're using an API (FastAPI for example), you'd do something like:
# """
#def predict_activities_via_api(df: pd.DataFrame) -> pd.DataFrame:
#    payload = {
#        "data": df.to_dict(orient="records")
#    }
#    response = requests.post("http://localhost:8000/predict", json=payload)
#    response.raise_for_status()
#    preds = response.json()["predictions"]  # depends on your API schema
#
#    results = df.copy()
#    results["predicted_activity"] = preds
#    return results
# """
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
        age = st.selectbox("Age", options=['18-29', '30-37', '38-52', '53+'])
    with col3:
        sex = st.selectbox("Sex", options=["Male", "Female"])

    st.markdown("### Upload accelerometer data")
    uploaded_file = st.file_uploader(
        "CSV or Parquet file (one participant, 24h data or selected window)",
        type=["csv", "parquet"]
    )

    submitted = st.form_submit_button("Run analysis")

# -----------------------------
# WHEN USER SUBMITS
# -----------------------------
if submitted:
    # Basic validation
    if not user_name:
        st.error("Please enter a name.")
        st.stop()

    if uploaded_file is None:
        st.error("Please upload a CSV file.")
        st.stop()

    # Read CSV or Parquet
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)

        else:
            st.error("Unsupported file format. Please upload a CSV or Parquet file.")
            st.stop()

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        st.error(
            f"The uploaded file is missing required columns: {missing_cols}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )
        st.write("Preview of your columns:", list(df.columns))
        st.stop()

    st.success("File uploaded successfully ✅")

    # Show basic info
    st.subheader("Participant & Data Overview")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            **Participant:** {user_name}
            **Age:** {age}
            **Sex:** {sex}
            """
        )
    with col_b:
        st.write("Data shape:", df.shape)
        st.write("Columns:", list(df.columns))

    with st.expander("Show first rows of uploaded data"):
        st.dataframe(df.head())

    # Run model
    with st.spinner("Running model and generating report..."):
        results = predict_activities(df)

    # -----------------------------
    # REPORT
    # -----------------------------
    st.markdown("---")
    st.header("Activity Prediction Report")

    # Summary: distribution of predicted activities
    activity_counts = results["predicted_activity"].value_counts().rename_axis("activity").reset_index(name="count")
    total = activity_counts["count"].sum()
    activity_counts["percentage"] = (activity_counts["count"] / total * 100).round(1)

    st.subheader("Predicted Activity Distribution")
    st.dataframe(activity_counts)

    # Simple bar chart
    st.bar_chart(
        data=activity_counts.set_index("activity")["count"],
        use_container_width=True
    )

    # Optional: time-series view (assuming timestamp is sortable)
    if "timestamp" in results.columns:
        st.subheader("Predicted Activity Over Time")
        # Ensure timestamp is datetime
        try:
            results["timestamp"] = pd.to_datetime(results["timestamp"])
            results_sorted = results.sort_values("timestamp")
            # Show only a sample if dataset is huge
            sample = results_sorted.iloc[:: max(1, len(results_sorted) // 500)]

            st.line_chart(
                data=sample.set_index("timestamp")[["acc_x", "acc_y", "acc_z"]],
                use_container_width=True
            )

            st.caption("Accelerometer signals over time (sampled).")
        except Exception:
            st.info("Timestamp column could not be parsed as datetime. Skipping time-series plot.")

    # Optional: show a sample of predictions
    st.subheader("Sample of Predicted Records")
    st.dataframe(results[["timestamp", "acc_x", "acc_y", "acc_z", "predicted_activity"]].head(20))

    # Textual summary
    st.subheader("Summary (for Demo Day)")
    main_activity = activity_counts.iloc[0]["activity"]
    main_pct = activity_counts.iloc[0]["percentage"]

    st.markdown(
        f"""
        - For **{user_name} ({age} years, {sex})**, the model predicted **{len(results)}** activity records.
        - The most frequent predicted activity was **{main_activity}**, representing **{main_pct}%** of the time window.
        - You can use the charts above during the demo to discuss daily patterns, model behavior and limitations.
        """
    )
