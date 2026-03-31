# Real-Time Transaction Fraud Detector

## Project Overview
This project implements a Machine Learning system that detects fraudulent credit card transactions in real time.

Fraud detection is an important problem in financial technology because fraudulent transactions are very rare compared to normal ones. This project demonstrates how Machine Learning algorithms can identify suspicious patterns.

## Problem Statement
Financial fraud causes billions of dollars in losses every year. Traditional rule-based systems fail to detect new fraud patterns.

This project builds a machine learning model that analyzes transaction behavior and identifies anomalies.

## Dataset
Credit Card Fraud Detection Dataset from Kaggle.

The dataset contains:
- Transaction Amount
- Time
- PCA-transformed features (V1–V28)
- Class label (0 = normal, 1 = fraud)

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Machine Learning (Logistic Regression)

## Machine Learning Method
The model uses **Supervised Learning – Classification**.

Algorithm Used:
Logistic Regression

Steps:
1. Load dataset
2. Data preprocessing
3. Feature scaling
4. Train-test split
5. Train ML model
6. Predict fraudulent transactions

## How to Run

Install dependencies

pip install pandas scikit-learn

Run the program

python fraud_detector.py

## Output
The system predicts whether a transaction is:

APPROVED – normal transaction  
DECLINED – suspicious transaction

## Author
Devansh Varshney
