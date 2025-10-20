#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diabetes Prediction System - Setup Test
Test environment and verify all components
"""

import os
import sys

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def check_python():
    """Check Python version"""
    print("✓ Python version: ", sys.version.split()[0])
    return True

def check_imports():
    """Check if required packages are installed"""
    print("\nChecking required packages...\n")
    
    packages = [
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'matplotlib',
        'seaborn',
        'jupyter',
        'joblib'
    ]
    
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (MISSING)")
            missing.append(pkg)
    
    return len(missing) == 0, missing

def check_files():
    """Check project files"""
    print("\nChecking project files...\n")
    
    files_needed = {
        'notebooks/01_EDA.ipynb': 'EDA Notebook',
        'notebooks/02_Model_Training.ipynb': 'Training',
        'notebooks/03_Model_Evaluation.ipynb': 'Evaluation',
        'notebooks/04_Demo_Prediction.ipynb': 'Demo',
        'src/preprocessing.py': 'Preprocessing module',
        'src/models.py': 'Models module',
        'src/evaluation.py': 'Evaluation module',
        'src/demo.py': 'Demo module',
    }
    
    missing = []
    for filepath, desc in files_needed.items():
        if os.path.exists(filepath):
            print(f"  ✓ {desc:<25} ({filepath})")
        else:
            print(f"  ✗ {desc:<25} (MISSING)")
            missing.append(filepath)
    
    return len(missing) == 0, missing

def check_data():
    """Check dataset"""
    print("\nChecking dataset...\n")
    
    if os.path.exists('data/diabetes.csv'):
        try:
            import pandas as pd
            df = pd.read_csv('data/diabetes.csv')
            print(f"  ✓ data/diabetes.csv found")
            print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"  ✗ Error reading data: {e}")
            return False
    else:
        print(f"  ✗ data/diabetes.csv NOT FOUND")
        print(f"\n  Download from Kaggle:")
        print(f"  → https://www.kaggle.com/uciml/pima-indians-diabetes-database")
        return False

def check_models():
    """Check trained models"""
    print("\nChecking trained models...\n")
    
    model_files = [
        'models/logistic_regression_model.pkl',
        'models/random_forest_model.pkl',
        'models/xgboost_model.pkl',
        'models/knn_model.pkl',
    ]
    
    found = 0
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"  ✓ {os.path.basename(model_file)}")
            found += 1
        else:
            print(f"  ⏳ {os.path.basename(model_file)} (not yet trained)")
    
    print(f"\n  Status: {found}/{len(model_files)} models available")
    return found > 0

def main():
    print_header("DIABETES PREDICTION SYSTEM - Setup Check")
    
    print("📋 Environment Check:")
    print("-" * 70)
    check_python()
    
    print_header("📦 Dependencies Check")
    deps_ok, missing_deps = check_imports()
    
    print_header("📁 Files Check")
    files_ok, missing_files = check_files()
    
    print_header("📊 Data Check")
    data_ok = check_data()
    
    print_header("🤖 Models Check")
    models_ok = check_models()
    
    # Summary
    print_header("📈 Setup Summary")
    
    print("Status:")
    print(f"  Environment ....... {'✅ READY' if check_python() else '❌ FAILED'}")
    print(f"  Dependencies ...... {'✅ OK' if deps_ok else '⚠️  MISSING: ' + ', '.join(missing_deps)}")
    print(f"  Project Files ..... {'✅ OK' if files_ok else '❌ MISSING'}")
    print(f"  Dataset ........... {'✅ READY' if data_ok else '⏳ DOWNLOAD NEEDED'}")
    print(f"  Trained Models .... {'✅ READY' if models_ok else '⏳ RUN NOTEBOOK 02'}")
    
    print("\n" + "="*70)
    
    if not deps_ok:
        print("\n⚠️  STEP 1: Install dependencies")
        print(f"   pip install {' '.join(missing_deps)}")
        print(f"   Or: pip install -r requirements.txt")
    
    if not data_ok:
        print("\n⚠️  STEP 2: Download dataset")
        print("   1. Go to: https://www.kaggle.com/uciml/pima-indians-diabetes-database")
        print("   2. Download diabetes.csv")
        print("   3. Place in: data/diabetes.csv")
    
    if files_ok and deps_ok and data_ok:
        print("\n✅ Ready to run!")
        print("   Start Jupyter: jupyter notebook")
        print("   Open: notebooks/01_EDA.ipynb")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)
