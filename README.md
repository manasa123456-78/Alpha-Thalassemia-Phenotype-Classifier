# Alpha-Thalassemia Phenotype Classifier
## Overview
This project presents a clinically inspired machine learning pipeline to classify alpha-thalassemia phenotypes using synthetically bootstrapped biomarker data.
It employs Gaussian noise perturbation to simulate biological variability and realistic laboratory measurement noise.
An XGBoost-based multi-class classifier is used for phenotype prediction with detailed explainability via SHAP values and model uncertainty estimation.

## Key Features
• Clinically validated biomarker ranges (MCV, MCH, HbA2, HBG, RBC)

• Synthetic data generation using bootstrapping + Gaussian noise

• Multi-class classification using XGBoost

• SHAP-based feature importance and per-sample decision plots

• PAC-learning and uncertainty-aware performance estimation

• Detailed evaluation: accuracy, per-class metrics, confusion matrix

## Project Structure

├── dataset/
│   └── alpha_thalassemia_bootstrapped_dataset.csv
├── images/
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   └── csv_head_image.png
├── src/
│   ├── data_generation.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── shap_analysis.py
├── report/
│   └── alpha_thalassemia_classifier_report.pdf
├── README.md
└── requirements.txt

## Methodology
### 1. Synthetic Data Generation
Created a 2,000-sample synthetic dataset based on peer-reviewed biomarker thresholds.

Bootstrapped sampling with Gaussian noise perturbation for biological realism.

Class distributions inspired by population-level prevalence studies.

### 2. Model Training
Multi-class XGBoost classifier trained on 1,500 samples.

Hyperparameter tuning for bias-variance tradeoff.

Stratified 500-sample test set.

### 3. Model Evaluation
Overall Accuracy: 98.8%

Class-wise precision, recall, and F1-score.

Confusion matrix analysis to visualize misclassification patterns.

### 4. Explainability & Uncertainty
SHAP plots for global feature importance.

Per-sample decision explanations.

Prediction confidence distribution plotted.

PAC-learning based real-world error bound estimation.

# Sample Results
### Classification Report

Accuracy: 98.8%

Class-wise F1-Score:
- Alpha-Thalassemia Major: 0.99
- Normal: 0.99
- Silent Carrier: 0.98
- Trait Carrier: 0.99
### Confusion Matrix

### SHAP Feature Importance

# Dataset Sample
Snapshot from the generated dataset:
