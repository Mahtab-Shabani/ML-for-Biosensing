# ML-for-Biosensing
This repository contains MATLAB scripts for unsupervised analysis of biosignals, specifically EEG and EMG, using time- and frequency-domain features with PCA and k-means clustering.

### Contents
**1. unsupervised_x1_x2.m:**

* detect EMG vs EEG using only x1 and x2 signals

**2. unsupervised_with_eeg.m:**

* same, plus binned EEG bands (delta, theta, alpha, beta).

**Features:** time-domain (MAV, RMS, ZCR…), spectral bands, spectral entropy, kurtosis.
**Analysis:** PCA, k-means clustering, signal-level EMG/EEG decision.
**Outputs:** .mat, .csv, and .png plots.

**Requirements:** MATLAB (compatible with R2014a and later)


**Note:** Results are saved as .mat and .csv files; plots are saved as .png.
