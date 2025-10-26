#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diabetes Prediction System - FAST Model Optimization
Huấn luyện nhanh với hyperparameter tuning
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
print("  DIABETES PREDICTION - FAST MODEL OPTIMIZATION")
print("  Mục tiêu: Đạt Accuracy >= 70%")
print("="*80 + "\n")

# ============================================================================
# 1. LẤY DỮ LIỆU
# ============================================================================
print("📊 BƯỚC 1: CHUẨN BỊ DỮ LIỆU")
print("-" * 80)

from src.preprocessing import DiabetesDataPreprocessor

preprocessor = DiabetesDataPreprocessor(random_state=42)
df = preprocessor.load_data('data/diabetes.csv')

print(f"✓ Dataset: {df.shape[0]} mẫu × {df.shape[1]} cột")
print(f"  Outcome: {(df['Outcome']==1).sum()} positive, {(df['Outcome']==0).sum()} negative")

df_clean = preprocessor.handle_missing_values(df)
X_train, X_test, y_train, y_test = preprocessor.prepare_data(df_clean, test_size=0.2)
feature_names = preprocessor.get_feature_names()
scaler = preprocessor.scaler

print(f"\n✓ Train: {X_train.shape[0]} mẫu | Test: {X_test.shape[0]} mẫu")

# ============================================================================
# 2. HUẤN LUYỆN NHANH
# ============================================================================
print("\n📈 BƯỚC 2: HUẤN LUYỆN NHANH")
print("-" * 80)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models = {}
results = {}

# ============================================================================
# 2.1 BEST KNN (thử k khác nhau)
# ============================================================================
print("\n🔹 1. K-NEAREST NEIGHBORS (tối ưu k)")

best_knn = None
best_knn_score = 0

for k in [3, 5, 7, 9, 11, 13]:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
    score = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy').mean()
    
    if score > best_knn_score:
        best_knn_score = score
        best_knn = knn

best_knn.fit(X_train, y_train)
knn_pred = best_knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
knn_auc = roc_auc_score(y_test, best_knn.predict_proba(X_test)[:, 1])

print(f"  ✓ Best k: {best_knn.n_neighbors}")
print(f"  ✓ Test Accuracy: {knn_acc:.4f}")
print(f"  ✓ ROC-AUC: {knn_auc:.4f}")

models['KNN'] = best_knn
results['KNN'] = knn_acc

# ============================================================================
# 2.2 BEST RANDOM FOREST
# ============================================================================
print("\n🔹 2. RANDOM FOREST (tối ưu)")

best_rf = None
best_rf_score = 0

for n_est in [100, 150, 200]:
    for max_d in [10, 15, 20]:
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, 
                                    random_state=42, n_jobs=-1)
        score = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy').mean()
        
        if score > best_rf_score:
            best_rf_score = score
            best_rf = rf

best_rf.fit(X_train, y_train)
rf_pred = best_rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])

print(f"  ✓ n_estimators: {best_rf.n_estimators}, max_depth: {best_rf.max_depth}")
print(f"  ✓ Test Accuracy: {rf_acc:.4f}")
print(f"  ✓ ROC-AUC: {rf_auc:.4f}")

models['Random Forest'] = best_rf
results['Random Forest'] = rf_acc

# ============================================================================
# 2.3 BEST XGBOOST
# ============================================================================
print("\n🔹 3. XGBOOST (tối ưu)")

best_xgb = None
best_xgb_score = 0

for n_est in [100, 150]:
    for max_d in [5, 7]:
        for lr in [0.05, 0.1]:
            xgb_model = xgb.XGBClassifier(n_estimators=n_est, max_depth=max_d, 
                                         learning_rate=lr, random_state=42, 
                                         eval_metric='logloss', verbosity=0)
            score = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy').mean()
            
            if score > best_xgb_score:
                best_xgb_score = score
                best_xgb = xgb_model

best_xgb.fit(X_train, y_train)
xgb_pred = best_xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:, 1])

print(f"  ✓ n_estimators: {best_xgb.n_estimators}, max_depth: {best_xgb.max_depth}, lr: {best_xgb.learning_rate}")
print(f"  ✓ Test Accuracy: {xgb_acc:.4f}")
print(f"  ✓ ROC-AUC: {xgb_auc:.4f}")

models['XGBoost'] = best_xgb
results['XGBoost'] = xgb_acc

# ============================================================================
# 2.4 GRADIENT BOOSTING
# ============================================================================
print("\n🔹 4. GRADIENT BOOSTING (tối ưu)")

best_gb = None
best_gb_score = 0

for n_est in [100, 150]:
    for lr in [0.05, 0.1]:
        gb = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, 
                                        max_depth=5, random_state=42)
        score = cross_val_score(gb, X_train, y_train, cv=5, scoring='accuracy').mean()
        
        if score > best_gb_score:
            best_gb_score = score
            best_gb = gb

best_gb.fit(X_train, y_train)
gb_pred = best_gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
gb_auc = roc_auc_score(y_test, best_gb.predict_proba(X_test)[:, 1])

print(f"  ✓ n_estimators: {best_gb.n_estimators}, learning_rate: {best_gb.learning_rate}")
print(f"  ✓ Test Accuracy: {gb_acc:.4f}")
print(f"  ✓ ROC-AUC: {gb_auc:.4f}")

models['Gradient Boosting'] = best_gb
results['Gradient Boosting'] = gb_acc

# ============================================================================
# 2.5 LOGISTIC REGRESSION
# ============================================================================
print("\n🔹 5. LOGISTIC REGRESSION")

lr = LogisticRegression(C=1, max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

print(f"  ✓ Test Accuracy: {lr_acc:.4f}")
print(f"  ✓ ROC-AUC: {lr_auc:.4f}")

models['Logistic Regression'] = lr
results['Logistic Regression'] = lr_acc

# ============================================================================
# 3. KẾT QUẢ
# ============================================================================
print("\n\n" + "="*80)
print("📊 KẾT QUẢ CUỐI CÙNG")
print("="*80)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("\nXếp hạng mô hình (Test Accuracy):\n")
for i, (model_name, accuracy) in enumerate(sorted_results, 1):
    status = "✅" if accuracy >= 0.70 else "⚠️ "
    print(f"  {i}. {status} {model_name:<25} {accuracy:.2%}")

# ============================================================================
# 4. LƯU MÔ HÌNH
# ============================================================================
print("\n" + "-"*80)
print("💾 LƯU MÔ HÌNH")
print("-"*80)

os.makedirs('models', exist_ok=True)

for model_name, model in models.items():
    model_file = f"models/{model_name.lower().replace(' ', '_')}_optimized.pkl"
    joblib.dump(model, model_file)
    print(f"  ✓ {model_name}: {model_file}")

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl')
print(f"  ✓ Scaler: models/scaler.pkl")
print(f"  ✓ Feature names: models/feature_names.pkl")

# ============================================================================
# 5. CHI TIẾT
# ============================================================================
print("\n" + "="*80)
print("📈 CHI TIẾT KẾT QUẢ")
print("="*80 + "\n")

print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
print("-" * 85)

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"{model_name:<25} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {roc_auc:<12.4f}")

# ============================================================================
# 6. KẾT LUẬN
# ============================================================================
print("\n" + "="*80)
print("✅ KIỂM ĐỊNH KẾT QUẢ")
print("="*80)

best_model_name = sorted_results[0][0]
best_accuracy = sorted_results[0][1]

print(f"\n🏆 Mô hình tốt nhất: {best_model_name}")
print(f"   Test Accuracy: {best_accuracy:.2%}")

if best_accuracy >= 0.70:
    print(f"\n   ✅ ĐẠT MỤC TIÊU >= 70%!")
else:
    print(f"\n   ⚠️  Kết quả thấp hơn 70%, nhưng đã tối ưu tốt nhất!")

print("\n💡 Lưu ý:")
print("   - Dữ liệu hiện tại chỉ có 100 mẫu (test set)")
print("   - Accuracy sẽ cao hơn khi sử dụng toàn bộ Kaggle dataset (768 mẫu)")
print("   - Để đạt 70%+ trên dataset nhỏ, cần thêm feature engineering")

print("\n" + "="*80)
print(f"Hoàn tất lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
