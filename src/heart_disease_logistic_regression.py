
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from pathlib import Path
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def print_summary(scores, title):
    print(f"\n{title} - Mean ± Std")
    for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
        mean = np.mean(scores[metric])
        std = np.std(scores[metric])
        name = metric.replace('test_', '').capitalize()
        print(f"{name:<10}: {mean:.3f} ± {std:.3f}")


# Load dataset
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "heart_disease_uci.csv"
df = pd.read_csv(DATA_PATH)


# Binary target
df['num'] = df['num'].apply(lambda x: -1 if x == 0 else 1)
print("Class distribution after binarization:")
print(df['num'].value_counts())

# Drop non-continuous features
drop_cols = ['sex', 'dataset', 'fbs', 'restecg', 'exang', 'slope', 'thal']
df = df.drop(columns=drop_cols)

X = df.drop(columns=['num'])
y = df['num']

categorical_feature = ['cp']
numerical_features = [col for col in X.columns if col not in categorical_feature]

# # One-Hot Encoding pipeline (with imputation to handle NaN)
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocess_onehot = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_feature)
])

model_onehot = Pipeline([
    ('preprocess', preprocess_onehot),
    ('classifier', LogisticRegression(max_iter=1000))
])


# Label Encoding pipeline
X_label = X.copy()
le = LabelEncoder()
X_label['cp'] = le.fit_transform(X_label['cp'])

preprocess_label = ColumnTransformer([
    ('num', num_pipeline, X_label.columns)
])


model_label = Pipeline([
    ('preprocess', preprocess_label),
    ('classifier', LogisticRegression(max_iter=1000))
])

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

scores_onehot = cross_validate(model_onehot, X, y, cv=kf, scoring=scoring)
print("One-Hot Encoding Results:", scores_onehot)
scores_label = cross_validate(model_label, X_label, y, cv=kf, scoring=scoring)
print("Label Encoding Results:", scores_label)
print("\n================ SUMMARY ================")
print_summary(scores_onehot, "One-Hot Encoding")
print_summary(scores_label, "Label Encoding")
print("=========================================\n")
