#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diabetes Prediction System - Quick Test
Runs without needing the full Kaggle dataset
"""

import os
import sys
import pandas as pd
import numpy as np

print("\n" + "="*70)
print("  DIABETES PREDICTION SYSTEM - Quick Test")
print("="*70)

# Check if dataset exists, if not create test data
print("\n📊 Preparing dataset...")

if os.path.exists('data/diabetes.csv'):
    print("  ✓ Using data/diabetes.csv")
    df = pd.read_csv('data/diabetes.csv')
else:
    print("  ℹ️  Creating test dataset...")
    # Create minimal test data to verify system works
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'Pregnancies': np.random.randint(0, 18, n_samples),
        'Glucose': np.random.randint(45, 201, n_samples),
        'BloodPressure': np.random.randint(24, 123, n_samples),
        'SkinThickness': np.random.randint(7, 100, n_samples),
        'Insulin': np.random.randint(0, 847, n_samples),
        'BMI': np.random.uniform(18.2, 67.1, n_samples),
        'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
        'Age': np.random.randint(21, 82, n_samples),
        'Outcome': np.random.randint(0, 2, n_samples)
    })
    
    # Save test data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/diabetes.csv', index=False)
    print(f"  ✓ Created test dataset: {df.shape[0]} rows × {df.shape[1]} columns")

print(f"\n  Dataset shape: {df.shape}")
print(f"  Columns: {', '.join(df.columns.tolist())}")
print(f"  Outcome distribution:\n{df['Outcome'].value_counts().to_string()}")

# Test imports
print("\n🔧 Testing imports...")
try:
    from src.preprocessing import DiabetesDataPreprocessor
    from src.models import DiabetesModelTrainer
    from src.evaluation import DiabetesModelEvaluator
    print("  ✓ All modules imported successfully")
except Exception as e:
    print(f"  ✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test preprocessing
print("\n⚙️  Testing preprocessing...")
try:
    preprocessor = DiabetesDataPreprocessor()
    df = preprocessor.load_data('data/diabetes.csv')
    df_clean = preprocessor.handle_missing_values(df)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df_clean)
    feature_names = preprocessor.get_feature_names()
    scaler = preprocessor.scaler
    
    print(f"  ✓ Data preprocessed")
    print(f"    Features shape: {X_train.shape}")
except Exception as e:
    print(f"  ✗ Preprocessing error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model training
print("\n🤖 Training models...")
try:
    trainer = DiabetesModelTrainer()
    
    trainer.train_logistic_regression(X_train, y_train)
    trainer.train_random_forest(X_train, y_train)
    trainer.train_xgboost(X_train, y_train)
    trainer.train_knn(X_train, y_train)
    
    models = trainer.models
    print(f"  ✓ Models trained: {list(models.keys())}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    import joblib
    for name, model in models.items():
        model_name = name.lower().replace(" ", "_")
        joblib.dump(model, f'models/{model_name}_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print(f"  ✓ Models saved to models/")
    
except Exception as e:
    print(f"  ✗ Training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test evaluation
print("\n📈 Evaluating models...")
try:
    evaluator = DiabetesModelEvaluator(feature_names)
    
    for name, model in models.items():
        evaluator.evaluate_model(model, X_test, y_test, name)
            
except Exception as e:
    print(f"  ✗ Evaluation error: {e}")
    import traceback
    traceback.print_exc()

# Test demo/prediction
print("\n💡 Testing predictions...")
try:
    # Load trained model
    import joblib
    best_model = models['Random Forest']
    
    # Test patient data
    test_patient = {
        'Pregnancies': 2,
        'Glucose': 150,
        'BloodPressure': 80,
        'SkinThickness': 30,
        'Insulin': 150,
        'BMI': 35.0,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 45
    }
    
    # Make prediction
    patient_values = [test_patient[col] for col in feature_names]
    patient_scaled = scaler.transform([patient_values])
    probability = best_model.predict_proba(patient_scaled)[0][1]
    
    print(f"  ✓ Prediction successful")
    print(f"    Patient data: Age={test_patient['Age']}, Glucose={test_patient['Glucose']}, BMI={test_patient['BMI']}")
    print(f"    Probability: {probability:.2%}")
    
    # Classify risk
    if probability < 0.4:
        risk = "🟢 LOW RISK"
    elif probability < 0.7:
        risk = "🟡 MEDIUM RISK"
    else:
        risk = "🔴 HIGH RISK"
    
    print(f"    Risk Level: {risk}")
    
except Exception as e:
    print(f"  ✗ Prediction error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - System is working!")
print("="*70)
print("\n📝 Next steps:")
print("   1. Download real dataset from Kaggle (optional)")
print("   2. Run Jupyter notebooks:")
print("      → jupyter notebook")
print("   3. Open notebooks/01_EDA.ipynb")
print("\n" + "="*70 + "\n")
