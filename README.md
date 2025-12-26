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


├── data/ # dataset (heart_disease_uci.csv)
├── src/ # source code
├── output.txt # real execution output
├── README.md
├── requirements.txt
└── .gitignore