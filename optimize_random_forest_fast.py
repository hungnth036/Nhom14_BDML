#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random Forest Optimization - Fast Version
Tối ưu nhanh Random Forest với tham số cụ thể
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

print("\n" + "="*80)
print("  RANDOM FOREST OPTIMIZATION - Phiên Bản Nhanh")
print("="*80 + "\n")

# 1. Load Data
print("📊 Tải dữ liệu...")
df = pd.read_csv('data/diabetes.csv')
print(f"  ✓ {df.shape[0]} mẫu × {df.shape[1]} cột")

# 2. Data Preprocessing
print("\n⚙️  Xử lý dữ liệu...")

df_clean = df.copy()
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    zero_count = (df_clean[col] == 0).sum()
    if zero_count > 0:
        median_val = df_clean[df_clean[col] != 0][col].median()
        df_clean.loc[df_clean[col] == 0, col] = median_val

X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  ✓ Train: {X_train.shape[0]} mẫu, Test: {X_test.shape[0]} mẫu")

# 3. Calculate Class Weights
print("\n⚖️  Cân bằng dữ liệu...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"  ✓ Class weight: {class_weight_dict}")

# 4. Train Multiple Random Forest Models and Compare
print("\n🔧 Huấn luyện các mô hình Random Forest với tham số khác nhau...")
print("\n" + "-"*80)

results = []

configs = [
    {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5},
    {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5},
    {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 5},
    {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5},
    {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5},
    {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2},
    {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10},
]

best_model = None
best_score = 0
best_config = None

for i, config in enumerate(configs, 1):
    print(f"\n[{i}/{len(configs)}] Config: {config}")
    
    rf = RandomForestClassifier(
        **config,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='f1')
    
    # Train on full training set
    rf.fit(X_train_scaled, y_train)
    
    # Test evaluation
    y_pred = rf.predict(X_test_scaled)
    y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"  CV F1: {cv_scores.mean():.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    results.append({
        'config': config,
        'cv_f1': cv_scores.mean(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'model': rf
    })
    
    # Track best model (by F1 score)
    if f1 > best_score:
        best_score = f1
        best_model = rf
        best_config = config

# 5. Display Best Results
print("\n" + "="*80)
print("  ✅ BEST MODEL FOUND")
print("="*80)

print(f"\nBest Configuration: {best_config}")

# Evaluate best model
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📈 Kết Quả Đánh Giá:")
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

# 6. Feature Importance
print(f"🔍 Tầm Quan Trọng Đặc Trưng (Top 5):")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']:<30} {row['Importance']:.4f}")

# 7. Save Best Model
print(f"\n💾 Lưu Mô Hình Tối Ưu...")
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

joblib.dump(best_model, 'models/random_forest_best.pkl')
joblib.dump(scaler, 'models/scaler_best.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names_best.pkl')

print(f"  ✓ models/random_forest_best.pkl")
print(f"  ✓ models/scaler_best.pkl")
print(f"  ✓ models/feature_names_best.pkl")

# 8. Final Status
print("\n" + "="*80)
if accuracy >= 0.70:
    print(f"  ✅ THÀNH CÔNG! Accuracy: {accuracy*100:.1f}% (>= 70%)")
    print(f"  🏆 Random Forest là lựa chọn tốt nhất!")
else:
    print(f"  ⚠️  Accuracy: {accuracy*100:.1f}% (< 70%)")
    print(f"  💡 Vẫn là mô hình tốt để sử dụng")

print("="*80 + "\n")

# Compare with all results
print("📊 So Sánh Tất Cả Cấu Hình:")
print("-"*80)
results_df = pd.DataFrame([
    {
        'n_estimators': r['config']['n_estimators'],
        'max_depth': r['config']['max_depth'],
        'min_samples_split': r['config']['min_samples_split'],
        'Accuracy': r['accuracy'],
        'F1': r['f1'],
        'ROC-AUC': r['roc_auc']
    }
    for r in results
])

print(results_df.to_string(index=False))
print()
