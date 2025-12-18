# Wearable Activity Recognition
***

This project presents an end-to-end wearable activity recognition system.

The solution includes:
- a complete data processing pipeline;
- two independent machine learning models;
- a REST API implemented with FastAPI;
- a front-end application built with Streamlit.

The input relies exclusively on tri-axial acceleration signals collected from a wrist-worn accelerometer, without using any additional user input—manual or sensor-based (e.g., heart rate or GPS).

Two complementary classification tasks are addressed:
- **i.** prediction of activity intensity levels using a Random Forest model;
- **ii.** prediction of specific physical activities using a Multilayer Perceptron (MLP).

*Data Source:* Training and evaluation data come from the [Capture-24](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001) dataset
(Chan Chang, S., et al. *Capture-24: Activity Tracker Dataset for Human Activity Recognition*. University of Oxford, 2021.)



## Background
This was the final project for the Data Science and AI bootcamp from Le Wagon Brasil.
It was developed by:

[Liana Bernat](https://github.com/LianaBernat)

[Renan Pereira](https://github.com/santossrenan)

[Renata Grassi](https://github.com/RenataGrassi)



## Data Preparation
### Why Preparation is Necessary

The Capture-24 dataset contains wrist-worn accelerometer data collected from 151 participants between 2014 and 2016 in the Oxfordshire region. Participants wore an Axivity AX3 device during daily living for approximately 24 hours. Activity labels were obtained using wearable cameras and sleep diaries, totaling more than 2,500 hours of annotated data.

The data are provided as one CSV file per participant, with a sampling interval of 0.01 s, resulting in an average of approximately 9 million rows per participant. This makes working directly with the raw time series computationally expensive and impractical for most modeling approaches.

In addition, raw acceleration values are not very informative on their own, but when processed over small time windows, they can reveal useful patterns for identifying different types of movement.

### Windowing (5-second windows)

The continuous acceleration signals were segmented into fixed, non-overlapping time windows of 5 seconds. This window length was chosen to avoid overly long segments while still capturing meaningful movement patterns.

Windows were discarded if they contained missing timestamps or accelerometer values (x, y, or z), fewer than 250 samples, or insufficient annotation coverage. Specifically, windows in which 50% or more of the activity annotations were missing were excluded to reduce label noise.

### Feature Extraction and Label Assignment

From each valid window, a set of features was extracted to summarize the acceleration signal:
- Time-domain features (mean, standard deviation, minimum, and maximum per axis)
- Energy-based features and mean acceleration magnitude
- Inter-axis correlations (x–y, x–z, y–z)
- Frequency-domain features extracted using FFT (Welch method applied to the magnitude signal)

Activity labels were assigned at the window level using majority voting over the annotations within each window. Fixed label encoding was applied, with ambiguous windows marked as −1.

Additional features were evaluated but discarded in the final versions of both models, as explained in the next section:
- Cyclic time features (hour sine and cosine)
- Participant metadata (ID, sex, age group)

The following fields were retained for auditing and traceability purposes:
- Window start and end timestamps
- Number of samples per window
- Window duration

### Processing in Chunks

Window processing was performed in non-overlapping chunks. Some observations at chunk boundaries were intentionally discarded to ensure that processing remained feasible in terms of memory usage and execution time.

## Feature Selection and Modeling Decisions
During model development, we evaluated the inclusion of participant sex, age group, and time-of-day information. Although these features increased overall accuracy, they degraded the F1-score of less frequent classes. Given the strong class imbalance, the final models rely exclusively on features derived from raw tri-axial acceleration signals.

## Models

Several machine learning models were evaluated, including Random Forest, XGBoost, and a Multilayer Perceptron neural network (MLP). Each model was tested on both labeling schemes: activity intensity classification (4 classes) and specific activity classification (10 classes).

Both classification tasks exhibit strong class imbalance; for example, activities such as sitting and sleeping are far more frequent than classes like sports or manual work.

### Random Forest — Activity Intensity (Walmsley2020)

Random Forest consistently outperformed MLP and XGBoost in predicting the four activity intensity levels defined according to Walmsley2020:
- Sleep
- Sedentary
- Light activity
- Moderate-to-vigorous activity

To mitigate class imbalance during training, SMOTE (Synthetic Minority Over-sampling Technique) was applied.
The model achieved an overall accuracy of approximately **~75%** and a macro F1-score of **~0.7**.

### MLP — Specific Activity (WillettsSpecific2018)

The Multilayer Perceptron (MLP) achieved better performance than Random Forest and XGBoost in predicting the ten activity classes defined according to WillettsSpecific2018:
- Bicycling
- Household chores
- Manual work
- Mixed activity
- Sitting
- Sleep
- Sports
- Standing
- Vehicle
- Walking

Class imbalance was addressed by using class weights computed with `class_weight="balanced"` during training.
The MLP achieved an overall accuracy of approximately **~60%**, with a macro F1-score of **~0.4**.

## Streamlit Application and Inference API

A Streamlit front-end application was developed to provide a simple interface for testing the activity recognition system. The application is available at the following URL: **https://wearableactivityrecognition.streamlit.app/**.

The application accepts a Parquet file containing high-frequency wrist-worn accelerometer data following the Capture-24 format. The file must include tri-axial acceleration signals (x, y, z) and timestamps, sampled at a high temporal resolution. To ensure valid inference, the file must contain a minimum number of valid observations per window.
If you do not have a compatible file, an example Parquet file is provided in the repository under **[`sample_data/example_participant.parquet`](sample_data/example_participant.parquet)**.

Once uploaded, the Streamlit application sends the data to a REST API deployed on Google Cloud Platform (GCP). The API applies the same preprocessing and feature extraction pipeline used during model training to ensure consistency between training and inference.

The API applies both trained models:
- a Random Forest model for activity intensity classification (4 classes), and
- a Multilayer Perceptron (MLP) for specific activity classification (10 classes).

Predictions are returned as structured JSON responses and consumed by the Streamlit application to generate visual summaries and daily activity reports.

## Future Improvements

- Apply a Leave-One-Participant-Out (LOPO) evaluation strategy to better assess model generalization across unseen participants.
- Explore deep learning approaches trained directly on raw acceleration signals, which would require more advanced data engineering and scalable preprocessing pipelines.
- Investigate alternative strategies for handling class imbalance, particularly for rare activities.
- Further tune existing models to improve performance on minority classes.


## Bibliography

Bao, L., & Intille, S. S. (2004). Activity recognition from user-annotated acceleration data. Proceedings of the 2nd International Conference on Pervasive Computing (Pervasive 2004).

Willetts M, Hollowell S, Aslett L, Holmes C, Doherty A. (2018) Statistical machine learning of sleep and physical activity phenotypes from sensor data in 96,220 UK Biobank participants. Scientific Reports. 8(1):7961.

Walmsley, R., Doherty, A., et al. (2022). Reallocation of time between device-measured movement behaviours and risk of incident cardiovascular disease. British Journal of Sports Medicine, 56(18), 1008-1015. DOI: 10.1136/bjsports-2021-105183.

## Data License and Attribution

This project uses the [Capture-24](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001) dataset provided by the University of Oxford.
(Chan Chang, S., et al. Capture-24: Activity Tracker Dataset for Human Activity Recognition. University of Oxford, 2021.)
The data were processed and transformed as part of this work.

The Capture-24 dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
https://creativecommons.org/licenses/by/4.0/

This project is not endorsed by the original data creators.
