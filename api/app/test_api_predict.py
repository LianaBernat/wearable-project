import requests

API_URL = "http://127.0.0.1:8000/predict"
FILE_PATH = "../data/data_processed/participants/P043_no_annotations.parquet"
OUTPUT_JSON = "../data/data_processed/participants/resultado_P043.json"

with open(FILE_PATH, "rb") as f:
    files = {"file": ("P043_no_annotations.parquet", f)}

    print("Enviando arquivo...")
    response = requests.post(API_URL, files=files)

print("Status:", response.status_code)

with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
    out.write(response.text)

print(f"JSON salvo em: {OUTPUT_JSON}")
