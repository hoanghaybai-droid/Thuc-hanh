# Verify app logic with XGBoost integration
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb

# Generate synthetic dataset for testing logic
np.random.seed(42)
n_samples = 500
df = pd.DataFrame({
    'DSRI': np.random.uniform(0.8, 2.0, n_samples),
    'GMI': np.random.uniform(0.8, 2.0, n_samples),
    'AQI': np.random.uniform(0.8, 2.0, n_samples),
    'SGI': np.random.uniform(0.8, 2.0, n_samples),
    'DEPI': np.random.uniform(0.8, 2.0, n_samples),
    'SGAI': np.random.uniform(0.8, 2.0, n_samples),
    'LVGI': np.random.uniform(0.8, 2.0, n_samples),
    'TATA': np.random.uniform(-0.2, 0.4, n_samples),
    'FRAUD_FLAG': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
})

X = df[['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']]
y = df['FRAUD_FLAG']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"[{name}] Accuracy: {accuracy_score(y_test, y_pred):.4f}, AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Test feature importances extraction
xgb_model = models['XGBoost']
importance_df = pd.DataFrame({
    'Chỉ số': X.columns,
    'Hệ số (Weight)': xgb_model.feature_importances_,
    'Chiều tác động': ['Tăng nguy cơ / Quan trọng' for _ in X.columns]
}).sort_values(by='Hệ số (Weight)', ascending=False)

print("\nXGBoost Feature Importances:")
print(importance_df)