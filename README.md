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

## Results (10-Fold Cross Validation)

The model was evaluated using 10-fold cross validation. Missing values were handled using `SimpleImputer` inside the pipeline to avoid data leakage.

| Encoding Method | Accuracy (Mean ± Std) | Precision (Mean ± Std) | Recall (Mean ± Std) | F1-score (Mean ± Std) |
|---|---:|---:|---:|---:|
| One-Hot Encoding | **0.809 ± 0.048** | 0.822 ± 0.053 | **0.833 ± 0.081** | **0.825 ± 0.055** |
| Label Encoding | 0.803 ± 0.047 | **0.822 ± 0.048** | 0.818 ± 0.090 | 0.818 ± 0.059 |

✅ One-Hot Encoding achieved slightly better overall performance, especially in Recall and F1-score, which are more important for correctly identifying patients with heart disease.
