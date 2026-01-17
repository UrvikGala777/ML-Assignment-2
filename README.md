# Breast Cancer Classification - End-to-End ML Project
## ML Assignment 2 | BITS Pilani

### a. Problem Statement
The objective of this project is to develop a machine learning solution to classify breast cancer tumors as either **Malignant (cancerous)** or **Benign (non-cancerous)** based on diagnostic measurements. Early and accurate diagnosis is crucial for effective treatment planning. This project compares six different classification algorithms to identify the best-performing model for this medical diagnostic task.

### [cite_start]b. Dataset Description [cite: 28, 30]
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset.
* **Source:** UCI Machine Learning Repository / Scikit-learn.
* **Instances:** 569 samples.
* **Features:** 30 numeric features (computed from a digitized image of a fine needle aspirate of a breast mass).
* **Classes:** Binary (Malignant, Benign).
* **Criteria Met:** Features (30 > 12) and Instances (569 > 500).

### c. Models Used & Comparison Table
The following models were implemented and evaluated:
1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbors (KNN)
4.  Naive Bayes (Gaussian)
5.  Random Forest (Ensemble)
6.  XGBoost (Ensemble)

#### Evaluation Metrics Comparison

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
|:--------------------|:---------|:------|:----------|:-------|:---------|:------|
| Logistic Regression | 0.974    | 0.997 | 0.972     | 0.986  | 0.979    | 0.944 |
| Decision Tree       | 0.947    | 0.944 | 0.957     | 0.957  | 0.957    | 0.887 |
| KNN                 | 0.947    | 0.982 | 0.958     | 0.958  | 0.958    | 0.888 |
| Naive Bayes         | 0.965    | 0.995 | 0.958     | 0.986  | 0.972    | 0.925 |
| Random Forest       | 0.965    | 0.995 | 0.958     | 0.986  | 0.972    | 0.925 |
| XGBoost             | 0.974    | 0.993 | 0.972     | 0.986  | 0.979    | 0.945 |

#### [cite_start]Observations [cite: 79]

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well due to the linear separability of many features in this dataset. It offers high accuracy and interpretability. |
| **Decision Tree** | Slightly lower accuracy compared to ensemble methods, likely due to overfitting on specific feature splits, though still robust (>90%). |
| **KNN** | Good performance (approx 96%), but computation time increases with dataset size. Scaling the data was critical for this model's success. |
| **Naive Bayes** | Surprisingly strong performance. The Gaussian assumption holds reasonably well for the dimensional features of the cells. |
| **Random Forest** | High stability and accuracy. As an ensemble method, it reduced the variance seen in the single Decision Tree model. |
| **XGBoost** | Competed for the top spot with Logistic Regression. Its boosting technique effectively handled hard-to-classify edge cases. |