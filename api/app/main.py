# api/main.py
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from tensorflow import keras
from joblib import load

from .preprocess_chunked import load_preprocessor, preprocess_parquet_chunked

from .config import (
    MODEL_RF_PATH,
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
# Loading models and preprocessor
# ---------------------------------------------------------

print("Loading models and preprocessor...")

#model_4 = keras.models.load_model(MLP_MODEL_4_PATH)
model_10 = keras.models.load_model(MLP_MODEL_10_PATH)
preprocessor, feature_names = load_preprocessor(
    MLP_PREPROCESSOR_PATH,
    MLP_FEATURE_NAMES_PATH
)

model_rf = load(MODEL_RF_PATH)


print("✅ Models and preprocessor loaded.")

# ---------------------------------------------------------
# Principal Endpoint: /predict
# ---------------------------------------------------------
@app.get("/")
async def main():
    return {'message': 'WEARABLE- PROJECT. API is running. Use the /predict endpoint to POST data.'}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives a .parquet file with columns ['time', 'x', 'y', 'z'],
    runs preprocessing in chunks, and returns predictions from 2 models:
    - Random Forest for 4-class model (Walmsley2020)
    - MLP for 10-class model (WillettsMET2018)
    """

    # --------------------------
    # 1. Read parquet
    # --------------------------
    if not file.filename.endswith(".parquet"):
        return JSONResponse(
            status_code=400,
            content={"error": "O arquivo deve ser .parquet"}
        )

    try:
        raw_bytes = await file.read()
        buffer = io.BytesIO(raw_bytes)
    except Exception as e:
        return JSONResponse(
        status_code=400,
        content={"error": f"Erro ao receber o arquivo: {e}"}
    )


    # --------------------------
    # 2. Preprocessor
    # --------------------------
    try:
         X_raw, X_ready, window_starts = preprocess_parquet_chunked(
            fileobj=buffer,
            preprocessor=preprocessor,
            feature_names=feature_names,
            batch_size=500_000
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
        preds_4 = model_rf.predict(X_raw)
        class_4 = preds_4

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
