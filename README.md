---
title: Laptop Price Predictor
emoji: 💻
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 💻 Laptop Price Predictor 🚀


A high-performance machine learning application designed to predict the price of laptops based on their hardware specifications. This tool provides instant estimations in Indian Rupees (₹), helping users understand the market value of various laptop configurations.

## 🌟 Key Features
- **Accurate Predictions**: Powered by a Random Forest Regressor optimized for price estimation.
- **Comprehensive Specs**: Considers Brand, Processor, RAM, SSD/HDD, OS, Graphics Card, Warranty, and more.
- **Interactive UI**: Clean and intuitive interface built with Gradio for seamless user interaction.
- **Automated Pipeline**: End-to-end data preprocessing and model inference pipeline.

## 🛠️ Tech Stack
- **Languages**: Python
- **Machine Learning**: Scikit-learn, NumPy, Pandas
- **Interface**: Gradio
- **Serialization**: Pickle

## 🚀 Getting Started

### 1. Installation
Ensure you have Python installed, then clone this repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training the Model
If you want to retrain the model with fresh data:
```bash
python train_model.py
```
This will process `laptop_data.csv` and generate an updated `model_pipeline.pkl`.

### 3. Running the Application
To launch the interactive prediction interface:
```bash
python app.py
```
Once running, open the local URL (usually `http://127.0.0.1:7860`) in your browser.

## 📊 Model Overview
The core of this project is a **Random Forest Regressor** pipeline. 

### Preprocessing
- **Categorical Encoding**: One-Hot Encoding for Brand, Processor, RAM Type, OS, etc.
- **Data Cleaning**: Automated extraction of numeric values from strings (e.g., converting "8 GB" to `8`).
- **Feature Scaling**: Handled within the scikit-learn pipeline for consistent results.

### Performance
The model is tuned with the following parameters:
- `max_depth`: 15
- `max_samples`: 0.85
- `random_state`: 42

## 📁 Project Structure
```text
├── laptop_data.csv       # Raw dataset
├── train_model.py        # Model training & evaluation script
├── app.py                # Main Gradio application
├── model_pipeline.pkl    # Serialized ML pipeline (Model + Preprocessor)
├── Laptop.ipynb          # Exploratory Data Analysis (EDA) notebook
└── requirements.txt      # Project dependencies
```

## 📝 License
This project is open-source and available for educational purposes.
