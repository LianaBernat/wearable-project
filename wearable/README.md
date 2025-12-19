# Wearable Activity Recognition — Modeling Pipeline

This repository contains the modeling components of a wearable activity
recognition pipeline built on the Capture-24 dataset.



## Project Overview

The pipeline is composed of two main stages:

### 1. Dataset Preparation

- Raw accelerometer signals are segmented into fixed 5-second windows.
- Feature engineering is applied to generate statistical, spectral, and contextual features.
- All participants are merged into a single modeling dataset.

### 2. Modeling

- Multiple machine learning models are trained on the final dataset:
  - **Random Forest** (baseline, classical machine learning)
  - **Multilayer Perceptron (MLP)** (deep learning)

The pipeline is structured to be modular and reproducible, with a clear
separation between data preparation, model definition, training, and inference.


## Dataset Preparation and Modeling Structure

```
dataset_preparation/
├── config.py # Data preparation configuration
├── annotation.py # Label treatment and validation
├── features.py # Feature extraction
├── participants_pipeline.py # Chunked processing per participant
├── join_chunks_per_participant.py # Logical consolidation per participant
├── join_participants.py # Merge all participants
└── README.md # Dataset preparation documentation

modeling/
├── MLP/
│ ├── config.py # Model and training configuration
│ ├── preprocessing.py # Feature preprocessing pipeline
│ ├── model.py # MLP architecture definition
│ └── train.py # Training routine
│
├── Random_forest/
│ ├── config.py # Random Forest configuration
│ ├── random_forest.py # Feature handling and model utilities
│ └── main.py # Training entry point
│

run_all.py # End-to-end pipeline orchestrator
```


## Training Environment

Model training was executed in different cloud environments depending on computational requirements:

- **Random Forest** models were trained on CPU using a virtual machine / managed job on **Google Vertex AI**.
- **MLP (deep learning)** models were trained in a **GPU-enabled environment (Google Colab)** due to their higher computational cost.

The local codebase reflects the full reproducible pipeline, but running the entire training process locally may not be feasible without sufficient computational resources.

All trained artifacts (models and preprocessors) are saved and later reused for inference in the API and Streamlit demo.



## Notes

- The API and demo applications **do not perform training**.
- Only pre-trained models and preprocessors are loaded for inference.
- The `run_all.py` script represents the full pipeline orchestration and serves as executable documentation of the project workflow.
