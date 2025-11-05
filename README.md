# LSTM Autoencoder for Time-Series Anomaly Detection

This project implements a **univariate LSTM Autoencoder** to detect anomalies in environmental time-series data from the Numenta Anomaly Benchmark (NAB) dataset.  

The model learns to reconstruct normal patterns and identifies anomalies when the reconstruction error crosses a dynamic threshold.

## Features
- Sequence normalization and sliding-window data generation
- LSTM encoder–decoder architecture using TensorFlow/Keras
- Dynamic percentile thresholding for adaptive anomaly detection
- Visualization of reconstruction errors and anomaly regions

##  Project Structure

- main.ipynb → Training and evaluation notebook
- data/ → Dataset (NAB ambient temperature CSV)
- models/ → Saved trained model file
- visuals/ → Generated plots


## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib  

## 📊 Dataset
[Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) — specifically the **ambient_temperature_system_failure.csv** file.

## ⚙️ Installation
Clone the repo and install dependencies:
```bash
pip install -r requirements.txt

