# api/main.py

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from tensorflow import keras
from joblib import load

from api.preprocess import load_preprocessor, preprocess_capture24

from api.config_old import (
    MLP_MODEL_4_PATH,
    MLP_MODEL_10_PATH,
    MLP_PREPROCESSOR_PATH,
    MLP_FEATURE_NAMES_PATH,
    WILLETS_LABELS,
    WALMSLEY_LABELS
)

# ---------------------------------------------------------
# API Setup
# ---------------------------------------------------------
app = FastAPI(
    title="WEARABLE-PROJECT API",
    description="API for preprocessing and activity prediction",
    version="1.0.0"
)

# ---------------------------------------------------------
# Loading artifacts
# ---------------------------------------------------------

print("Loading models and preprocessor...")

model_4 = keras.models.load_model(MLP_MODEL_4_PATH)
model_10 = keras.models.load_model(MLP_MODEL_10_PATH)
preprocessor, feature_names = load_preprocessor(
    MLP_PREPROCESSOR_PATH,
    MLP_FEATURE_NAMES_PATH
)

print("✅ Models and preprocessor loaded.")

# ---------------------------------------------------------
# Principal Endpoint: /predict
# ---------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives a CSV with columns ['time', 'x', 'y', 'z'],
    runs preprocessing, and returns predictions from 2 models:
    - 4-class model (Walmsley2020)
    - 10-class model (WillettsMET2018)
    """

    # --------------------------
    # 1. Read CSV
    # --------------------------
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Error reading CSV: {str(e)}"}
        )

    # --------------------------
    # 2. Preprocessor
    # --------------------------
    try:
        X_ready, window_starts = preprocess_capture24(
            df,
            preprocessor=preprocessor,
            feature_names=feature_names
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Error in preprocessing: {str(e)}"}
        )

    # --------------------------
    # 3. Predicting with 2 models
    # --------------------------
    try:
        # modelo de 10 classes
        preds_10 = model_10.predict(X_ready)
        class_10 = preds_10.argmax(axis=1)

        # modelo de 4 classes
        preds_4 = model_4.predict(X_ready)
        class_4 = preds_4.argmax(axis=1)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error running the model: {str(e)}"}
        )

    # --------------------------
    # 4. Json answer
    # --------------------------
    willets_timeline = []
    walmsley_timeline = []

    for ts, c10, c4 in zip(window_starts, class_10, class_4):

        willets_timeline.append({
            "window_start": ts.isoformat(),
            "label_id": int(c10),
            "label": WILLETS_LABELS[int(c10)]
        })

        walmsley_timeline.append({
            "window_start": ts.isoformat(),
            "label_id": int(c4),
            "label": WALMSLEY_LABELS[int(c4)]
        })

    return {
        "willetts_10classes": willets_timeline,
        "walmsley_4classes": walmsley_timeline
    }
