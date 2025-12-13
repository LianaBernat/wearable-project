import requests

#API_URL = "http://127.0.0.1:8000/predict"
API_URL = "https://wearable-api-1009461955584.us-central1.run.app/predict"
FILE_PATH = "../data/data_processed/participants/P043_no_annotations_small.parquet"
OUTPUT_JSON = "../data/data_processed/participants/resultado_P043_small.json"

with open(FILE_PATH, "rb") as f:
    files = {"file": ("P043_no_annotations_small.parquet", f)}

    print("Enviando arquivo...")
    response = requests.post(API_URL, files=files)

print("Status:", response.status_code)

with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
    out.write(response.text)

print(f"JSON salvo em: {OUTPUT_JSON}")
