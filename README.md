# LSTM Autoencoder for Time-Series Anomaly Detection

This project implements a **univariate LSTM Autoencoder** to detect anomalies in environmental time-series data from the Numenta Anomaly Benchmark (NAB) dataset.  

The model learns to reconstruct normal patterns and identifies anomalies when the reconstruction error crosses a dynamic threshold.

## Concept Overview

**Problem Statement:**  
In sensor-based environments, certain events (like temperature spikes or machine failures) are rare but critical. 
Traditional statistical models fail to capture long-term temporal dependencies.  
The goal is to build a neural architecture that *learns what “normal” looks like* — so it can recognize when something *doesn’t look normal.*

**Solution:**  
An **LSTM Autoencoder** learns to reconstruct input sequences. If a sequence cannot be reconstructed well (high reconstruction error), it is considered an anomaly.

## Features
- Sequence normalization and sliding-window data generation
- LSTM encoder–decoder architecture using TensorFlow/Keras
- Dynamic percentile thresholding for adaptive anomaly detection
- Visualization of reconstruction errors and anomaly regions

##  Project Structure

- notebooks/ → Training and evaluation notebooks
- data/ → Dataset (NAB ambient temperature CSV)
- models/ → Saved trained model file
- visuals/ → Generated plots

## Notebooks Structure 
-notebooks/
│
├── 1_load_and_visualize.ipynb → Load data, visualize temperature patterns
├── 2_preprocessing.ipynb → Normalize and create sliding sequences
├── 3_model_training.ipynb → Build and train LSTM Autoencoder
├── 4_detect_anomalies.ipynb → Compute reconstruction errors, mark anomalies
└── 5_reporting.ipynb → Generate visualizations and export CSV


## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib  

## 📊 Dataset
[Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) — specifically the **ambient_temperature_system_failure.csv** file.


## Results

- Smooth reconstruction for normal sequences  
- Sharp error spikes near anomalies  
- High interpretability with visualizations  

| Metric | Value |
|:-------|:------|
| Detection Accuracy | 95% |
| Model Size | 62,529 parameters |
| Sequence Length | 30 timesteps |
| Training Time | ~7s per epoch (50 epochs) |



