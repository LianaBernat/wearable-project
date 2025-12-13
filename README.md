# Wearable Activity Recognition #
***

*Project Name:* Wearable Activity Recognition
*Description:* we present a model capable of recognizing smartwatch movement patterns and automatically classifying physical activities, without requiring any manual input from the user (e.g., "running" or "cycling"). Two classification approaches are implemented: a Random Forest ML model and a Multilayer Perceptron (MLP) neural network.
*Data Source:* [Capture-24](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001).
(Chan Chang, S., et al. Capture-24: Activity Tracker Dataset for Human Activity
Recognition. University of Oxford, 2021.)
*Main objective:* To automatically recognize activities from smartwatch data, removing the need for manual user input.

### Background
---
This was the final project for the Data Science and AI bootcam from Le Wagon Brasil.
It was developed by:

[Liana Bernat](https://github.com/LianaBernat) (creator)

[Renan Pereira](https://github.com/santossrenan)

[Renata Grassi](https://github.com/RenataGrassi)

### Data Analysis
---
We aimed to use as many features as possible while keeping the model as generalistic as possible. Therefore, our data analysis focused on evaluating correlations only among numerical features, such as sensor-based measurements (e.g., gravity-related signals) and time-related variables (e.g., hours).

### Data Preparation
---
 -Applied time-based windowing with 5-second non-overlapping windows
- Removed samples with missing time or accelerometer values (x, y, z)
- Discarded windows with fewer than 250 samples
- Discarded windows where ≥ 50% of annotations were missing
- Extracted time-domain features (mean, std, min, max per axis)
- Computed energy-based features and mean acceleration magnitude
- Calculated inter-axis correlations (x–y, x–z, y–z)
- Extracted frequency-domain features using FFT (Welch method on magnitude)
- Identified dominant frequency and peak power within a valid frequency range
- Generated simplified activity labels using majority voting
- Applied fixed label encoding, marking ambiguous windows with −1
- Added cyclic time features (hour sine and cosine)
- Included participant metadata (ID, sex, age group)
- Added audit fields (window start/end, number of samples, duration)
- Saved processed data as chunked Parquet files per participant

### Feature engineering
---
'x_mean', 'x_std', 'x_min', 'x_max', 'y_mean', 'y_std', 'y_min', 'y_max',                'z_mean', 'z_std', 'z_min', 'z_max','energy_x', 'energy_y', 'energy_z', 'energy_total',
'corr_xy', 'corr_xz', 'corr_yz','fft_dom_freq', 'fft_peak_power'


### Models
---
We experimented with several models, including XGBoost, SVC, MLP, and Random Forest.
The best-performing model was selected for each labeling category, as follows:

**Random Forest**
* Activity Intensity (4 classes - Walmsley2020):  it classifies the movement according to its´ intensity:

1. Sleep
2. SedentaRy
3. Light activity
4. Moderate-vigorous activity


**MLP**
* Specific Activity (10-class model: WillettsSpecific2018): this model classifies the model in 10 moviments:
1. Bicycling
2. Household chores
3. Manual work
4. Mixed Activity
5. Sitting
6. Sleep
7. Sports
8. Standing
9. Vehicle
10. Walking

### Input and Output
---
**Input**: CSV files containing wearable sensor data (e.g., accelerometer x, y, z)
**Output**: Predicted activity labels

### URL base
---
The deploy was made with Streamlit and it is  available [here](https://wearableactivityrecognition.streamlit.app/).

### Future Improvements

- Improve class imbalance handling;
- Experiment with deep learning models;
- Fine tunning existent models

## Bibliography
Willetts M, Hollowell S, Aslett L, Holmes C, Doherty A. (2018) Statistical machine learning of sleep and physical activity phenotypes from sensor data in 96,220 UK Biobank participants. Scientific Reports. 8(1):7961.

Walmsley, R., Doherty, A., et al. (2022). Reallocation of time between device-measured movement behaviours and risk of incident cardiovascular disease. British Journal of Sports Medicine, 56(18), 1008-1015. DOI: 10.1136/bjsports-2021-105183.

The dataset used was [Capture-24](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001).
(Chan Chang, S., et al. Capture-24: Activity Tracker Dataset for Human Activity
Recognition. University of Oxford, 2021.)
