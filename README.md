# ML-for-Biosensing
This repository contains MATLAB scripts for unsupervised analysis of biosignals, specifically EEG and EMG, using time- and frequency-domain features with PCA and k-means clustering.

### Contents
**1. unsupervised_x1_x2.m:**

* detect EMG vs EEG using only x1 and x2 signals

<img width="500"  alt="unsupervised_x1_x2_pca" src="https://github.com/user-attachments/assets/66eb5c7c-311d-4c37-bdc2-b6f3e7953bae" />
<img width="500" alt="unsupervised_x1_x2_bp_hi" src="https://github.com/user-attachments/assets/fb7fa2dd-5828-4d49-b3df-5ed0104f8f9f" />


**2. unsupervised_with_eeg.m:**

* same, plus binned EEG bands (delta, theta, alpha, beta).

<img width="500" alt="unsupervised_with_eeg_pca" src="https://github.com/user-attachments/assets/3ba88017-3399-40c4-8c4b-8a3f87c19a2c" />
<img width="500" alt="unsupervised_with_eeg_bp_hi" src="https://github.com/user-attachments/assets/6385d278-96eb-400c-aaed-cfc715cb1c11" />

**Features:** time-domain (MAV, RMS, ZCR…), spectral bands, spectral entropy, kurtosis.
**Analysis:** PCA, k-means clustering, signal-level EMG/EEG decision.
**Outputs:** .mat, .csv, and .png plots.

**Requirements:** MATLAB (compatible with R2014a and later)


**Note:** Results are saved as .mat and .csv files; plots are saved as .png.








