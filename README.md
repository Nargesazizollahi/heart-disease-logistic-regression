# Heart Disease Prediction using Logistic Regression

This repository contains a machine learning project for predicting the presence of heart disease
using Logistic Regression.

---

## 📊 Dataset

The dataset is obtained from Kaggle:  
[Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

The target variable `num` is converted into a binary classification problem:
- `0`: No heart disease
- `1`: Presence of heart disease

---

## 🧹 Preprocessing

- Non-continuous features were removed as specified in the project description.
- The categorical feature `cp` was encoded using:
  - One-Hot Encoding
  - Label Encoding

Two separate datasets were created and evaluated.

---

## 🤖 Model

- **Algorithm:** Logistic Regression  
- **Evaluation Method:** 10-Fold Cross Validation  
- **Metrics:**  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  

---

## 📈 Results

Experimental results show that **One-Hot Encoding outperforms Label Encoding**
for this dataset due to the non-ordinal nature of the categorical feature.

---

## 🚀 How to Run

1. Install required packages:

```bash
pip install -r requirements.txt
