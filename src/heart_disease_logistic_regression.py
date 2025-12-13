
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("heart_disease_uci.csv")

# Binary target
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

# Drop non-continuous features
drop_cols = ['sex', 'dataset', 'fbs', 'restecg', 'exang', 'slope', 'thal']
df = df.drop(columns=drop_cols)

X = df.drop(columns=['num'])
y = df['num']

categorical_feature = ['cp']
numerical_features = [col for col in X.columns if col not in categorical_feature]

# One-Hot Encoding pipeline
preprocess_onehot = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
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
    ('num', StandardScaler(), X_label.columns)
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
scores_label = cross_validate(model_label, X_label, y, cv=kf, scoring=scoring)

print("One-Hot Encoding Results:", scores_onehot)
print("Label Encoding Results:", scores_label)
