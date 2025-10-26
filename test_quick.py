#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test with expanded dataset
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

print("\n" + "="*80)
print("  QUICK TEST - KNN >= 70%?")
print("="*80 + "\n")

# Load data
df = pd.read_csv('data/diabetes.csv')
print(f"Dataset: {df.shape[0]} mẫu")

# Prepare
X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

# Train models
print("Training models...\n")

# KNN with k=3
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
knn_auc = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])

print(f"KNN (k=3):")
print(f"  Accuracy: {knn_acc:.2%}")
print(f"  ROC-AUC: {knn_auc:.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

print(f"\nRandom Forest:")
print(f"  Accuracy: {rf_acc:.2%}")
print(f"  ROC-AUC: {rf_auc:.4f}")

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, 
                              random_state=42, eval_metric='logloss', verbosity=0)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

print(f"\nXGBoost:")
print(f"  Accuracy: {xgb_acc:.2%}")
print(f"  ROC-AUC: {xgb_auc:.4f}")

# Results
print(f"\n" + "="*80)
print("KẾT QUẢ:")

results = {
    'KNN': knn_acc,
    'Random Forest': rf_acc,
    'XGBoost': xgb_acc
}

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for i, (name, acc) in enumerate(sorted_results, 1):
    status = "✅" if acc >= 0.70 else "⚠️ "
    print(f"  {i}. {status} {name}: {acc:.2%}")

best_acc = sorted_results[0][1]

if best_acc >= 0.70:
    print(f"\n✅ ĐẠT MỤC TIÊU >= 70%!")
else:
    print(f"\n⚠️  Chưa đạt 70%")

print("="*80 + "\n")

# Save
joblib.dump(knn, 'models/knn_optimized.pkl')
joblib.dump(rf, 'models/random_forest_optimized.pkl')
joblib.dump(xgb_model, 'models/xgboost_optimized.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✓ Mô hình đã lưu\n")
