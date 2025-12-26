# Heart Disease Prediction using Logistic Regression

This repository presents a complete machine learning pipeline for predicting the presence of heart disease using **Logistic Regression**.

Two encoding strategies were investigated for the categorical feature `cp`:
- ✅ **One-Hot Encoding**
- ✅ **Label Encoding**

📌 **Raw execution output (fold-by-fold + summary)** is available in: **`output.txt`**  
📌 **Clean results table** is also available in: **`results.md`** (recommended for quick review)

---

## 📊 Dataset

Source (Kaggle):  
https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

Target variable `num` was converted into a binary classification task:
- `0` → No heart disease
- `1` → Presence of heart disease (originally 1–4)

---

## 📁 Project Structure

```text
heart-disease-logistic-regression/
├── data/        # dataset (heart_disease_uci.csv)
├── src/         # source code
│   ├── heart_disease_logistic_regression.py
│   └── plot_results.py
├── figures/     # generated plots (comparison charts)
├── output.txt   # raw real execution output
├── results.md   # clean results table (markdown)
├── README.md
├── requirements.txt
└── .gitignore
