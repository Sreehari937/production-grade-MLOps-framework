# debug_check.py  — save this in phase3_automation/ and run it
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

base = "C:/phase 5 pipeline MLOps/phase2_model"
train = pd.read_parquet(f"{base}/churn_train_v1.parquet")
val   = pd.read_parquet(f"{base}/churn_val_v1.parquet")
test  = pd.read_parquet(f"{base}/churn_test_v1.parquet")

print("Train columns:", train.columns.tolist())
print("Train shape:", train.shape)

target = [c for c in train.columns if "churn" in c.lower()][0]
print(f"\nTarget: '{target}'")
print("Target distribution:\n", train[target].value_counts())
print("Target dtype:", train[target].dtype)

drop_cols = ["date_of_registration","customer_id","pincode","city","state","telecom_partner","gender"]

def clean(df):
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.select_dtypes(include=["number"])
    df = df.fillna(df.median())
    return df.reset_index(drop=True)

X_train = clean(train.drop(columns=[target]))
y_train = train[target].reset_index(drop=True)
X_test  = clean(test.drop(columns=[target]))
y_test  = test[target].reset_index(drop=True)

print(f"\nX_train shape: {X_train.shape}")
print(f"Columns: {X_train.columns.tolist()}")
print(f"y_train unique: {sorted(y_train.unique().tolist())}")
print(f"y_train balance: {y_train.value_counts().to_dict()}")

model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
model.fit(X_train, y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
print(f"\nTest AUC: {auc:.4f}")

# add this to the bottom of debug_check.py and rerun

import numpy as np

# Feature importances
print("\nFeature importances:")
for name, imp in sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")

# Correlation of each feature with churn
print("\nCorrelation with churn:")
for col in X_train.columns:
    corr = X_train[col].corr(y_train.astype(float))
    print(f"  {col}: {corr:.4f}")

# Churn rate by tenure quartile (sanity check)
print("\nChurn rate by tenure_days quartile:")
df_check = X_train.copy()
df_check['churn'] = y_train
df_check['tenure_q'] = pd.qcut(df_check['tenure_days'], 4)
print(df_check.groupby('tenure_q')['churn'].mean())

# add to debug_check.py and run
train['total_activity'] = train['calls_made'] + train['sms_sent'] + train['data_used']
train['avg_data_per_day'] = train['data_used'] / (train['tenure_days'] + 1)
train['engagement_score'] = 0.5*train['calls_made'] + 0.3*train['sms_sent'] + 0.2*train['data_used']

print("Correlations with churn:")
for col in ['total_activity', 'avg_data_per_day', 'engagement_score']:
    print(f"  {col}: {train[col].corr(train['churn'].astype(float)):.4f}")