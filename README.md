# Titanic Survival Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![status](https://img.shields.io/badge/status-completed-green)
![project](https://img.shields.io/badge/project-machine--learning-blue)


## Overview

This project implements a complete machine learning pipeline to predict passenger survival on the Titanic using structured passenger data. The workflow includes Exploratory Data Analysis (EDA), data preprocessing, feature engineering, model training, hyperparameter tuning, and model evaluation.

Multiple classification models are compared, including Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree. The best-performing model is selected based on ROC AUC, Accuracy, and F1-Score.

This project demonstrates best practices in machine learning pipeline design using scikit-learn.

## Objective

The goal of this project is to build a classification model that predicts whether a passenger survived the Titanic disaster based on demographic and travel features.

This is a binary classification problem:

* 0 → Did not survive
* 1 → Survived

## Dataset Description

The dataset contains passenger information including demographics, ticket details, and travel class.

Each row represents one passenger.

### Dataset Columns

| Column      | Description                       | Type        |
| ----------- | --------------------------------- | ----------- |
| PassengerId | Unique passenger identifier       | Numerical   |
| Survived    | Survival status (Target variable) | Binary      |
| Pclass      | Passenger class (1st, 2nd, 3rd)   | Categorical |
| Name        | Passenger name                    | Text        |
| Sex         | Passenger gender                  | Categorical |
| Age         | Passenger age                     | Numerical   |
| SibSp       | Number of siblings/spouses aboard | Numerical   |
| Parch       | Number of parents/children aboard | Numerical   |
| Ticket      | Ticket number                     | Text        |
| Fare        | Ticket fare                       | Numerical   |
| Cabin       | Cabin number                      | Categorical |
| Embarked    | Port of embarkation (C, Q, S)     | Categorical |


## Features Used

### Numerical Features

* Age
* Fare
* SibSp
* Parch

### Categorical Features

* Sex
* Pclass
* Embarked

## Features Removed

The following features were dropped:

* PassengerId → Identifier only
* Name → Unstructured text
* Ticket → Mostly unique values
* Cabin → Too many missing values

## Machine Learning Pipeline

The project uses a structured preprocessing pipeline to ensure consistency and prevent data leakage.

### Numerical preprocessing

* Missing value imputation using median
* Feature scaling using StandardScaler

### Categorical preprocessing

* Missing value imputation using most frequent value
* Encoding using OneHotEncoder

All preprocessing is implemented using:

* Pipeline
* ColumnTransformer

## Models Implemented

Three classification models were trained and evaluated:

### 1. Logistic Regression

Baseline linear classification model.

Advantages:

* Fast
* Interpretable
* Good baseline performance

### 2. K-Nearest Neighbors (KNN)

Distance-based classification model.

Hyperparameter tuning was performed using the elbow method to find the optimal number of neighbors.

Best parameter:

k = 5

### 3. Decision Tree

Tree-based classification model.

Two versions were tested:

Unconstrained Tree
Pruned Tree (max_depth = 5)

The pruned tree showed better generalization and reduced overfitting.

## Model Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC AUC Score

ROC curves were also generated to compare classification performance.

## Best Model

The best model is:

KNN (Optimized)

Performance:

* ROC AUC: 0.8478
* Accuracy: 0.8212
* F1-Score: 0.7500

This model provides the best balance between classification performance and generalization.

## Key Insights

From Exploratory Data Analysis:

* Female passengers had significantly higher survival rates
* Higher passenger class increased survival probability
* Passengers with higher fares were more likely to survive
* Sex and Pclass were the most important predictors

## Files Generated

classification_model_comparison.csv
Model performance comparison table

roc_curve_comparison.png
ROC curve comparison between models

knn_accuracy_vs_k.png
KNN hyperparameter tuning plot

report.md
Final analysis and model selection summary

## Project Structure

```
project/
│
├── Titanic-Dataset.csv
├── notebook.ipynb
├── classification_model_comparison.csv
├── roc_curve_comparison.png
├── knn_accuracy_vs_k.png
├── report.md
└── README.md
```

## Technologies Used

Python

Libraries:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run the notebook

```bash
jupyter notebook
```

Open the notebook and run all cells.

## Machine Learning Concepts Demonstrated

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Feature engineering
* Pipeline design
* Hyperparameter tuning
* Overfitting and pruning
* Model evaluation
* ROC curve analysis
* Model comparison

## Author

Nadine Octavia
Machine Learning Project — Titanic Survival Prediction
