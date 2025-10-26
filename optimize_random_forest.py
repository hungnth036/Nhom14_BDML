#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diabetes Prediction System - Random Forest Optimization
Tối ưu hóa Random Forest để đạt >70% Accuracy
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

print("\n" + "="*80)
print("  RANDOM FOREST OPTIMIZATION - Tối ưu hóa Random Forest")
print("="*80 + "\n")

# 1. Load Data
print("📊 Bước 1: Tải dữ liệu...")
try:
    df = pd.read_csv('data/diabetes.csv')
    print(f"  ✓ Tải thành công: {df.shape[0]} mẫu × {df.shape[1]} cột")
except Exception as e:
    print(f"  ✗ Lỗi: {e}")
    exit(1)

# 2. Data Preprocessing
print("\n⚙️  Bước 2: Xử lý dữ liệu...")

# Handle missing values (zeros)
df_clean = df.copy()
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    zero_count = (df_clean[col] == 0).sum()
    if zero_count > 0:
        median_val = df_clean[df_clean[col] != 0][col].median()
        df_clean.loc[df_clean[col] == 0, col] = median_val
        print(f"  ✓ {col}: thay {zero_count} giá trị 0 bằng {median_val:.2f}")

# Separate features and target
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  ✓ Train: {X_train.shape[0]} mẫu, Test: {X_test.shape[0]} mẫu")
print(f"  ✓ Train - Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
print(f"  ✓ Test  - Class 0: {(y_test==0).sum()}, Class 1: {(y_test==1).sum()}")

# 3. Calculate Class Weights (để cân bằng dataset)
print("\n⚖️  Bước 3: Cân bằng dữ liệu...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"  ✓ Class weight 0: {class_weight_dict[0]:.2f}")
print(f"  ✓ Class weight 1: {class_weight_dict[1]:.2f}")

# 4. Hyperparameter Tuning with GridSearchCV
print("\n🔧 Bước 4: Tối ưu hyperparameters (GridSearchCV)...")
print("  ⏳ Đang tìm kiếm tham số tối ưu...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

grid_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_scaled, y_train)

print(f"\n  ✓ Tìm kiếm hoàn tất!")
print(f"  ✓ Best F1-Score (CV): {grid_search.best_score_:.4f}")
print(f"  ✓ Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"     - {param}: {value}")

# 5. Train Best Model
print("\n🤖 Bước 5: Huấn luyện mô hình tối ưu...")
best_rf = grid_search.best_estimator_

# Cross-validation on best model
cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"  ✓ 5-Fold CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 6. Evaluate on Test Set
print("\n📈 Bước 6: Đánh giá trên tập test...")

y_pred = best_rf.predict(X_test_scaled)
y_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"  ✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  ✓ Precision: {precision:.4f}")
print(f"  ✓ Recall:    {recall:.4f}")
print(f"  ✓ F1-Score:  {f1:.4f}")
print(f"  ✓ ROC-AUC:   {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n  Confusion Matrix:")
print(f"    TN: {cm[0,0]:<5} FP: {cm[0,1]:<5}")
print(f"    FN: {cm[1,0]:<5} TP: {cm[1,1]:<5}")

# Classification Report
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# 7. Feature Importance
print("\n🔍 Bước 7: Tầm quan trọng của các đặc trưng...")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']:<30} {row['Importance']:.4f}")

# 8. Save Model
print("\n💾 Bước 8: Lưu mô hình tối ưu...")
os.makedirs('models', exist_ok=True)

joblib.dump(best_rf, 'models/random_forest_optimized.pkl')
joblib.dump(scaler, 'models/scaler_optimized.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names_optimized.pkl')

print(f"  ✓ models/random_forest_optimized.pkl")
print(f"  ✓ models/scaler_optimized.pkl")
print(f"  ✓ models/feature_names_optimized.pkl")

# 9. Summary
print("\n" + "="*80)
if accuracy >= 0.70:
    print(f"  ✅ THÀNH CÔNG! Accuracy: {accuracy*100:.1f}% (>= 70%)")
else:
    print(f"  ⚠️  Accuracy: {accuracy*100:.1f}% (< 70%)")
print("="*80 + "\n")

# 10. Save Results
results = {
    'Best Parameters': grid_search.best_params_,
    'Test Metrics': {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
    },
    'CV Score': cv_scores.mean(),
    'Feature Importance': feature_importance.to_dict()
}

import json
with open('results/random_forest_optimization_results.json', 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError

    json.dump(results, f, indent=2, default=convert)
    print("✓ Kết quả lưu vào: results/random_forest_optimization_results.json\n")
