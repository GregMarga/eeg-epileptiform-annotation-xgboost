# EEG Epileptiform Detection – XGBoost Baseline

This repository contains a simple and interpretable baseline pipeline for
epileptiform activity detection from EEG recordings, using handcrafted
time-domain features and an XGBoost classifier evaluated with a strict
Leave-One-Patient-Out (LOPO) protocol.

The code is organized as a clear, step-by-step processing pipeline, starting
from raw EDF files and ending with patient-independent classification results.

---

## Repository Structure

- `preprocessing.py`  
  EEG preprocessing utilities and window extraction around annotations.

- `batch_make_windows.py`  
  Batch processing of EDF files into annotation-centered EEG windows.

- `batch_extract_features_from_npz_basic.py`  
  Extraction of handcrafted time-domain features from EEG windows.

- `classification_evaluation.py`  
  Leave-One-Patient-Out (LOPO) evaluation using an XGBoost classifier.

---

## EEG Preprocessing

For each EDF recording, the following preprocessing steps are applied:

1. **Channel renaming**  
   EEG channel names are cleaned and standardized.

2. **Average re-referencing**  
   Signals are re-referenced to the common average.

3. **Automatic bad-channel detection and interpolation**  
   Channels with abnormally high or low variance are automatically detected
   and interpolated.

4. **Band-pass filtering**  
   A band-pass FIR filter in the range **0.5–40 Hz** is applied.

---

## Window Extraction

- EEG windows are extracted **centered on expert annotations**.
- For each annotation, a window of **250 ms before and 250 ms after** the
  annotation onset is used.
- This results in fixed-length, annotation-centered EEG epochs.

For each `.edf` file, a corresponding `.npz` file containing all extracted
windows and labels is created.

### Excluded Recordings
Recordings from patients **p28** and file p58_GHB_M1681_0000033[2] of **P58** were excluded from further
processing, as they followed a different format and contained a large number
of artifacts, making them incompatible with the rest of the dataset.

---

## Feature Extraction

Each window-based `.npz` file
is converted into a feature-based representation.

For each EEG channel, the following **time-domain features** are extracted:

- Zero-crossing count  
- Number of local maxima  
- Number of local minima  
- Root Mean Square (RMS) amplitude  
- Skewness  
- Kurtosis 

The resulting feature vectors are saved in new `.npz` files, one per original
EDF recording.

---

## Classification and Evaluation

Classification is performed using an **XGBoost** binary classifier.

### Evaluation Protocol
- A **Leave-One-Patient-Out (LOPO)** scheme is used.
- In each fold, all recordings from one patient are held out for testing,
  while the model is trained on all remaining patients.
- This ensures a **strict patient-independent evaluation**, avoiding data
  leakage.

### Model Details
- XGBoost with shallow decision trees.
- Class imbalance is handled via `scale_pos_weight`.
- Performance is reported using:
  - Accuracy
  - Balanced Accuracy
  - Sensitivity
  - Specificity
  - F1-score
  - ROC-AUC
  - PR-AUC

This setup serves as a strong and interpretable baseline for comparison with
more advanced or patient-specific models.

---


## Notes
- No patient-specific tuning or fine-tuning is applied.
- The focus of this repository is on clarity, reproducibility, and
  patient-independent evaluation.

---

## Installation

```bash
pip install -r requirements.txt
