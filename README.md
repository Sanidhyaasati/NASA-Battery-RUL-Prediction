# NASA Li-ion Battery Remaining Useful Life (RUL) Prediction

This repository contains a comprehensive Data Science and Machine Learning pipeline designed to predict the State-of-Health (SoH) and Remaining Useful Life (RUL) of NASA Lithium-ion batteries. 

Battery degradation is a highly complex, non-linear phenomenon. In critical engineering applications (such as aerospace and electric vehicles), accurately predicting when a battery will fail is essential for safety. 

## 🔋 Project Overview

This project tackles battery degradation analysis in two primary phases:
1. **Binary Classification (State of Health):** Determining whether a battery is currently "Healthy" (1) or "Degraded" (0) based on real-time telemetry (cycle number, capacity, and temperature).
2. **Time-Series Regression (RUL):** Forecasting the exact number of charge/discharge cycles remaining before the battery crosses its End-of-Life (EOL) threshold.

## 🛠️ Methodology & Pipeline

### 1. Data Engineering & Preprocessing
* Extracted **2,769 discharge cycles** from raw NASA battery datasets.
* Dynamically calculated the End-of-Life (EOL) threshold for every unique battery (defined as reaching 75% of its initial starting capacity).
* Engineered advanced domain-specific features to capture memory-based degradation: `Cumulative_Ah`, `Capacity_Loss`, and `Capacity_Fade_Rate`.

### 2. Anomaly Detection
* Utilized an **Isolation Forest** model to detect anomalous capacity drops.
* Successfully flagged early-onset degradation in 19 out of 34 batteries with a **mean early detection lead time of 62 cycles**.

### 3. Classification & Handling Imbalanced Data
* Evaluated 8 baseline Classical Machine Learning models (Logistic Regression, SVM, KNN, ANN, Naive Bayes, SGD, Decision Tree, Random Forest) to classify the battery's current state.
* **The Problem:** The dataset was highly imbalanced (1628 Healthy vs 205 Degraded training samples), causing baseline models to ignore the minority class (e.g., initial Random Forest had a degraded recall of just `0.10`).
* **The Solution:** Implemented **SMOTE (Synthetic Minority Over-sampling Technique)** strictly on the training set, balancing the classes to 1628-1628. This drastically improved the Random Forest's recall for degraded batteries from `0.10` to `0.77`.
* Tuned hyperparameters using `GridSearchCV` to optimize the $K$-Nearest Neighbors (Best Accuracy: **90%**) and Random Forest (Accuracy: **81%** with high safety recall).

### 4. Time-Series Forecasting (Deep Learning vs. XGBoost)
* **Long Short-Term Memory (LSTM) Neural Network:** 
  * Reformatted chronological data into 10-cycle sliding windows, forcing the model to rely on short-term capacity memory to predict future capacity fade.
  * Achieved an impressive test **RMSE of 0.2017** on normalized data.
* **XGBoost Regression:** 
  * Trained an XGBoost model leveraging the engineered features. 
  * Feature importance analysis revealed that `Capacity_Loss` (31.7%) and `Ambient_Temperature` (29.2%) were the strongest predictors of RUL.
  * The optimized XGBoost model achieved an **RMSE of 29.26 cycles** and an **MAE of 18.39 cycles** when predicting the exact Remaining Useful Life.

## 📁 Repository Structure

* `Project_MSE643.ipynb`: The core Jupyter Notebook containing all data extraction, EDA, modeling, and evaluation.
* `metadata.csv`: The aggregated metadata extracted from the raw NASA datasets containing cycle information, temperature, and capacity.
* `requirements.txt`: Python package dependencies required to run the notebook.

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Sanidhyaasati/NASA-Battery-RUL-Prediction.git
