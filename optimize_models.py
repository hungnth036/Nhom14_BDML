#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diabetes Prediction System - Advanced Model Training & Optimization
Huấn luyện với tối ưu hóa siêu tham số (Hyperparameter Tuning)
Mục tiêu: Đạt Accuracy >= 70%
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("  DIABETES PREDICTION - ADVANCED TRAINING WITH HYPERPARAMETER TUNING")
print("  Mục tiêu: Đạt Accuracy >= 70%")
print("="*80 + "\n")

# ============================================================================
# 1. LẤY DỮ LIỆU
# ============================================================================
print("📊 BƯỚC 1: CHUẨN BỊ DỮ LIỆU")
print("-" * 80)

from src.preprocessing import DiabetesDataPreprocessor

try:
    preprocessor = DiabetesDataPreprocessor(random_state=42)
    df = preprocessor.load_data('data/diabetes.csv')
    
    print(f"✓ Dataset: {df.shape[0]} mẫu × {df.shape[1]} cột")
    print(f"  Outcome: {(df['Outcome']==1).sum()} positive, {(df['Outcome']==0).sum()} negative")
    
    # Xử lý missing values
    df_clean = preprocessor.handle_missing_values(df)
    
    # Chia train/test
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df_clean, test_size=0.2)
    feature_names = preprocessor.get_feature_names()
    scaler = preprocessor.scaler
    
    print(f"\n✓ Train set: {X_train.shape[0]} mẫu")
    print(f"✓ Test set: {X_test.shape[0]} mẫu")
    print(f"✓ Features: {len(feature_names)}\n")
    
except Exception as e:
    print(f"✗ Lỗi chuẩn bị dữ liệu: {e}")
    sys.exit(1)

# ============================================================================
# 2. HYPERPARAMETER TUNING
# ============================================================================
print("\n📈 BƯỚC 2: HUẤN LUYỆN VỚI HYPERPARAMETER TUNING")
print("-" * 80)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_validate

# Dictionary lưu kết quả
results = {}
best_models = {}

# ============================================================================
# 2.1 LOGISTIC REGRESSION
# ============================================================================
print("\n🔹 1. LOGISTIC REGRESSION")
print("  Tuning: C (regularization), max_iter")

lr_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'max_iter': [1000, 2000]
}

try:
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42, solver='lbfgs'),
        lr_params,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    lr_grid.fit(X_train, y_train)
    
    best_lr = lr_grid.best_estimator_
    lr_train_score = best_lr.score(X_train, y_train)
    lr_test_score = best_lr.score(X_test, y_test)
    
    print(f"  ✓ Best params: C={lr_grid.best_params_['C']}, max_iter={lr_grid.best_params_['max_iter']}")
    print(f"  ✓ Train Accuracy: {lr_train_score:.4f}")
    print(f"  ✓ Test Accuracy: {lr_test_score:.4f}")
    
    best_models['Logistic Regression'] = best_lr
    results['Logistic Regression'] = lr_test_score
    
except Exception as e:
    print(f"  ✗ Lỗi: {e}")

# ============================================================================
# 2.2 K-NEAREST NEIGHBORS (Tối ưu k)
# ============================================================================
print("\n🔹 2. K-NEAREST NEIGHBORS")
print("  Tuning: n_neighbors, weights, algorithm")

knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree']
}

try:
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_params,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    knn_grid.fit(X_train, y_train)
    
    best_knn = knn_grid.best_estimator_
    knn_train_score = best_knn.score(X_train, y_train)
    knn_test_score = best_knn.score(X_test, y_test)
    
    print(f"  ✓ Best params: n_neighbors={knn_grid.best_params_['n_neighbors']}, "
          f"weights={knn_grid.best_params_['weights']}")
    print(f"  ✓ Train Accuracy: {knn_train_score:.4f}")
    print(f"  ✓ Test Accuracy: {knn_test_score:.4f}")
    
    best_models['KNN'] = best_knn
    results['KNN'] = knn_test_score
    
except Exception as e:
    print(f"  ✗ Lỗi: {e}")

# ============================================================================
# 2.3 RANDOM FOREST
# ============================================================================
print("\n🔹 3. RANDOM FOREST")
print("  Tuning: n_estimators, max_depth, min_samples_split")

rf_params = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10]
}

try:
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_params,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    rf_grid.fit(X_train, y_train)
    
    best_rf = rf_grid.best_estimator_
    rf_train_score = best_rf.score(X_train, y_train)
    rf_test_score = best_rf.score(X_test, y_test)
    
    print(f"  ✓ Best params: n_estimators={rf_grid.best_params_['n_estimators']}, "
          f"max_depth={rf_grid.best_params_['max_depth']}")
    print(f"  ✓ Train Accuracy: {rf_train_score:.4f}")
    print(f"  ✓ Test Accuracy: {rf_test_score:.4f}")
    
    best_models['Random Forest'] = best_rf
    results['Random Forest'] = rf_test_score
    
except Exception as e:
    print(f"  ✗ Lỗi: {e}")

# ============================================================================
# 2.4 XGBOOST
# ============================================================================
print("\n🔹 4. XGBOOST")
print("  Tuning: n_estimators, max_depth, learning_rate")

xgb_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}

try:
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        xgb_params,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    xgb_grid.fit(X_train, y_train)
    
    best_xgb = xgb_grid.best_estimator_
    xgb_train_score = best_xgb.score(X_train, y_train)
    xgb_test_score = best_xgb.score(X_test, y_test)
    
    print(f"  ✓ Best params: n_estimators={xgb_grid.best_params_['n_estimators']}, "
          f"max_depth={xgb_grid.best_params_['max_depth']}, "
          f"learning_rate={xgb_grid.best_params_['learning_rate']}")
    print(f"  ✓ Train Accuracy: {xgb_train_score:.4f}")
    print(f"  ✓ Test Accuracy: {xgb_test_score:.4f}")
    
    best_models['XGBoost'] = best_xgb
    results['XGBoost'] = xgb_test_score
    
except Exception as e:
    print(f"  ✗ Lỗi: {e}")

# ============================================================================
# 2.5 GRADIENT BOOSTING
# ============================================================================
print("\n🔹 5. GRADIENT BOOSTING")
print("  Tuning: n_estimators, learning_rate, max_depth")

gb_params = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

try:
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    gb_grid.fit(X_train, y_train)
    
    best_gb = gb_grid.best_estimator_
    gb_train_score = best_gb.score(X_train, y_train)
    gb_test_score = best_gb.score(X_test, y_test)
    
    print(f"  ✓ Best params: n_estimators={gb_grid.best_params_['n_estimators']}, "
          f"learning_rate={gb_grid.best_params_['learning_rate']}")
    print(f"  ✓ Train Accuracy: {gb_train_score:.4f}")
    print(f"  ✓ Test Accuracy: {gb_test_score:.4f}")
    
    best_models['Gradient Boosting'] = best_gb
    results['Gradient Boosting'] = gb_test_score
    
except Exception as e:
    print(f"  ✗ Lỗi: {e}")

# ============================================================================
# 3. KẾT QUẢ & LƯU MÔ HÌNH
# ============================================================================
print("\n\n" + "="*80)
print("📊 KẾT QUẢ CUỐI CÙNG")
print("="*80)

# Sắp xếp theo accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("\nXếp hạng mô hình (Test Accuracy):\n")
for i, (model_name, accuracy) in enumerate(sorted_results, 1):
    status = "✅" if accuracy >= 0.70 else "⚠️ "
    print(f"  {i}. {status} {model_name:<25} {accuracy:.2%}")

# Lưu mô hình tốt nhất
print("\n" + "-"*80)
print("💾 LƯU MÔ HÌNH")
print("-"*80)

os.makedirs('models', exist_ok=True)

for model_name, model in best_models.items():
    model_file = f"models/{model_name.lower().replace(' ', '_')}_optimized.pkl"
    joblib.dump(model, model_file)
    print(f"  ✓ {model_name}: {model_file}")

# Lưu scaler và feature names
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl')
print(f"  ✓ Scaler: models/scaler.pkl")
print(f"  ✓ Feature names: models/feature_names.pkl")

# ============================================================================
# 4. TẠO REPORT
# ============================================================================
print("\n" + "="*80)
print("📈 BÁO CÁO CHI TIẾT")
print("="*80 + "\n")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
print("-" * 85)

for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"{model_name:<25} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {roc_auc:<12.4f}")

# ============================================================================
# 5. KIỂM ĐỊNH TRÊN TEST SET
# ============================================================================
print("\n" + "="*80)
print("✅ KIỂM ĐỊNH MÔ HÌNH TRÊN TEST SET")
print("="*80)

best_model_name = sorted_results[0][0]
best_model = best_models[best_model_name]
best_accuracy = sorted_results[0][1]

print(f"\n🏆 Mô hình tốt nhất: {best_model_name}")
print(f"   Test Accuracy: {best_accuracy:.2%}")

if best_accuracy >= 0.70:
    print(f"\n   ✅ ĐẠT MỤC TIÊU >= 70%!")
else:
    print(f"\n   ⚠️  Chưa đạt 70%, nhưng đã tối ưu!")

print("\n" + "="*80)
print(f"Hoàn tất lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
