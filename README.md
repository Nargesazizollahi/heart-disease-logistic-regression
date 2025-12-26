# Heart Disease Prediction using Logistic Regression

This repository contains a machine learning project for predicting the presence of heart disease using **Logistic Regression**.
Two encoding strategies were compared for the categorical feature `cp`:
- **One-Hot Encoding**
- **Label Encoding**

✅ The full real execution output is available in **`output.txt`**.

---

## 📊 Dataset

Dataset source (Kaggle):  
https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

The target variable `num` was converted into a binary classification problem:
- `0` → No heart disease  
- `1` → Presence of heart disease (originally 1–4)

---
## 📁 Project Structure

```text
heart-disease-logistic-regression/
├── data/        # dataset (heart_disease_uci.csv)
├── src/         # source code
├── output.txt   # real execution output
├── README.md
├── requirements.txt
└── .gitignore

## 📈 Results (10-Fold Cross Validation)

The model was evaluated using 10-fold cross validation. Missing values were handled using `SimpleImputer` inside the pipeline to avoid data leakage.

| Encoding Method | Accuracy (Mean ± Std) | Precision (Mean ± Std) | Recall (Mean ± Std) | F1-score (Mean ± Std) |
|---|---:|---:|---:|---:|
| One-Hot Encoding | **0.809 ± 0.048** | 0.822 ± 0.053 | **0.833 ± 0.081** | **0.825 ± 0.055** |
| Label Encoding | 0.803 ± 0.047 | **0.822 ± 0.048** | 0.818 ± 0.090 | 0.818 ± 0.059 |

✅ **Conclusion:** One-Hot Encoding achieved slightly better overall performance, especially in Recall and F1-score, which are more important for correctly identifying patients with heart disease.
