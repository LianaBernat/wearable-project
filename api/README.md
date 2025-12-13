# WEARABLE-PROJECT API

This directory contains the REST API used in the Wearable Project (Capture-24).

The API:

- Preprocesses accelerometer `.parquet` files in chunks
- Extracts features in 5-second windows
- Runs activity prediction using:
  - MLP (10-class) model
  - Random Forest (4-class) model
- Returns timelines in JSON format

---
## 1. Project structure

```text
api/
│
├── Dockerfile
├── requirements.txt
├── __init__.py
│
└── app/
    ├── __init__.py
    ├── main.py
    ├── config.py
    ├── preprocess_chunked.py
    ├── features.py
    └── model/
        ├── mlp_baseline_10classes.keras
        ├── randomforest.joblib
        ├── preprocessor.joblib
        └── feature_names.joblib
```

## 2. Running the API locally (development)

### 2.1. Install dependencies
From inside the ```api/ folder```:

### 2.1. Install dependencies

Still inside the ```api/ folder```:
```bash
pip install -r requirements.txt
```

### 2.2. Start the API server
From inside the ```api/ folder```:

```bash
uvicorn app.main:app --reload
```
The API will be available at:

- http://127.0.0.1:8000

- http://127.0.0.1:8000/docs
 (Swagger UI)

- http://127.0.0.1:8000/redoc

## 3. Using Docker
From inside the ```api/ folder```:

```bash
docker build -t wearable-api .
docker run -d -p 8000:8000 wearable-api
```

The API will be available at:
- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs

*Note:*
*With -d option, the container runs in detached mode (in the background).*

To stop the container, find its ID with:
```bash
docker ps
```
To get the containers logs, use:
```bash
docker logs <container_id>
```
Then stop it with:
```bash
docker stop <container_id>
```

## 4. Endpoints
- GET/ : Health check
- POST/ predict : Upload accelerometer `.parquet` file with columsns time, x, y,z and get activity predictions in JSON
Processing:
1. Reads the file in chunks (to support large datasets).
2. Builds 5-second windows.
3. Extracts numerical features.
4. Applies preprocessing:
  - Normalized features for the MLP (10-class).
  - Raw features for the Random Forest (4-class).
5. Returns JSON with two timelines:
  - Willetts 10-class model
  - Walmsley 4-class model
Output (simplified structure):
```json
{
  "willetts_10classes": [
    {
      "window_start": "YYYY-MM-DDTHH:MM:SS",
      "label_id": 0,
      "label": "walking"
    },
    ...
  ],
  "walmsley_4classes": [
    {
      "window_start": "YYYY-MM-DDTHH:MM:SS",
      "label_id": 2,
      "label": "sedentary"
    },
    ...
  ]
}
```

## 5. Notes
- Only .parquet input is supported.
- The preprocessing pipeline is optimized for large files (millions of rows).
- Model and preprocessor paths are configured in app/config.py.
- Trained model files are located in app/model/.
