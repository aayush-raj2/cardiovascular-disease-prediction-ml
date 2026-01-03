# Cardiovascular Disease Prediction using Machine Learning

## Overview
This project predicts the presence of cardiovascular disease using various machine learning algorithms. The system analyzes patient health data and assists in early risk detection.

## Dataset
- Source: Cardiovascular Disease Dataset
- Records: 70,000+
- Target Variable: `cardio` (0 = No disease, 1 = Disease)

## Algorithms Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

## Methodology
1. Data preprocessing and feature scaling
2. Exploratory Data Analysis and visualization
3. Correlation analysis
4. Model training and evaluation
5. Model comparison and final selection

## Results
Random Forest achieved the highest accuracy and balanced precision-recall performance, making it suitable for cardiovascular disease risk prediction.

## How to Run
```bash
pip install -r requirements.txt
python3 cardio_ml.py
```

## Disclaimer 
This project is for educational purposes only and does not replace professional medical diagnosis.
