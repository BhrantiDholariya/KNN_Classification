# Project 2 — Data Classification Using AI 
**DecodeLabs Industrial Training | Batch 2026 | Artificial Intelligence**

---

## Overview

A Data Classification model built using Python and Scikit-learn as part of the DecodeLabs AI Engineering internship. This project demonstrates the complete supervised learning pipeline — loading a dataset, exploring it, splitting it, training a KNN model, and evaluating its accuracy — using the classic Iris flower dataset.

---

## Main Functions & Concepts Used

### 1. `load_iris()` — Loading the Dataset
Loads the built-in Iris flower dataset from scikit-learn. Contains 150 samples across 3 classes (Setosa, Versicolor, Virginica) with 4 features each.

### 2. `dataset.keys()` / `.shape` / `.DESCR` — Exploring the Dataset
Used to understand the structure of the dataset — checking keys, number of rows and columns, feature names, top 5 rows, and the full dataset description.

### 3. `dataset.data` / `dataset.target` — Separating Features and Labels
`dataset.data` holds the input features (X) and `dataset.target` holds the output labels (y) — separating what the model learns from and what it predicts.

### 4. `train_test_split()` — Splitting the Data
Splits the dataset into **70% training** and **30% testing** sets. The `random_state=42` parameter ensures the split is reproducible every time the code runs.

### 5. `KNeighborsClassifier(n_neighbors=5)` — The KNN Algorithm
Initializes the K-Nearest Neighbors classifier with K=5. Classifies a new data point by looking at the 5 closest points in the training data and taking a majority vote.

### 6. `.fit()` — Training the Model
Trains the KNN model on the training data. The model memorizes the pattern map of features to labels.

### 7. `.predict()` — Making Predictions
Applies the trained model on the test data and generates predicted class labels for each test sample.

### 8. `accuracy_score()` — Evaluating the Model
Compares the predicted labels against the actual labels and returns the percentage of correctly classified samples.

---

*Project 2 Completed | DecodeLabs AI Internship 2026*
