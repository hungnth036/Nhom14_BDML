#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diabetes Prediction System - Project Check
Version: 1.0 | Status: Complete | Date: 20/10/2025
"""

import os
import sys

def print_banner():
    """Print project banner"""
    print("\n" + "="*70)
    print("DIABETES PREDICTION SYSTEM - Project Status Check")
    print("="*70 + "\n")

def check_files():
    """Check required files"""
    print("Checking required files...\n")
    
    required_files = {
        'README.md': 'Main documentation',
        'requirements.txt': 'Dependencies list',
        'notebooks/01_EDA.ipynb': 'EDA Notebook',
        'notebooks/02_Model_Training.ipynb': 'Training Notebook',
        'notebooks/03_Model_Evaluation.ipynb': 'Evaluation Notebook',
        'notebooks/04_Demo_Prediction.ipynb': 'Demo Notebook',
        'src/preprocessing.py': 'Preprocessing Module',
        'src/models.py': 'Models Module',
        'src/evaluation.py': 'Evaluation Module',
        'src/demo.py': 'Demo Module',
    }
    
    all_exist = True
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print("[OK] " + description + " ... " + file_path)
        else:
            print("[MISSING] " + description + " ... " + file_path)
            all_exist = False
    
    print()
    return all_exist

def check_data():
    """Check data files"""
    print("Checking data files...\n")
    
    if os.path.exists('data/diabetes.csv'):
        print("[OK] data/diabetes.csv found")
        return True
    else:
        print("[MISSING] data/diabetes.csv")
        print("Download from: https://www.kaggle.com/uciml/pima-indians-diabetes-database")
        print()
        return False

def check_models():
    """Check trained models"""
    print("Checking trained models...\n")
    
    model_files = [
        'models/logistic_regression_model.pkl',
        'models/random_forest_model.pkl',
        'models/xgboost_model.pkl',
        'models/knn_model.pkl',
        'models/scaler.pkl',
        'models/feature_names.pkl'
    ]
    
    found_count = 0
    for model_file in model_files:
        if os.path.exists(model_file):
            print("[OK] " + model_file)
            found_count += 1
        else:
            print("[PENDING] " + model_file)
    
    print("\nStatus: {}/{} models created\n".format(found_count, len(model_files)))
    
    if found_count == 0:
        print("Run: notebooks/02_Model_Training.ipynb to create models\n")
    
    return found_count == len(model_files)

def print_quick_start():
    \"\"\"In hướng dẫn nhanh\"\"\"
    print(\"\\n\" + \"=\"*70)
    print(\"🚀 HƯỚNG DẪN NHANH (Quick Start)\")
    print(\"=\"*70 + \"\\n\")
    
    print(\"📝 BƯỚC 1: Chuẩn Bị (5 phút)\")
    print(\"-\" * 70)
    print(\"1. Tải dữ liệu từ Kaggle:\")
    print(\"   → https://www.kaggle.com/uciml/pima-indians-diabetes-database\")
    print(\"   → Đặt file diabetes.csv vào: final/data/\\n\")\n    \n    print(\"2. Cài thư viện:\")\n    print(\"   → pip install -r requirements.txt\\n\")\n    \n    print(\"📝 BƯỚC 2: Chạy Phân Tích (20 phút)\")
    print(\"-\" * 70)
    print(\"1. Khởi động Jupyter:\")\n    print(\"   → jupyter notebook\\n\")\n    \n    print(\"2. Chạy 4 notebooks theo thứ tự:\")\n    print(\"   → 01_EDA.ipynb (Phân tích dữ liệu) [5 phút]\")\n    print(\"   → 02_Model_Training.ipynb (Huấn luyện) [10 phút]\")\n    print(\"   → 03_Model_Evaluation.ipynb (Đánh giá) [3 phút]\")\n    print(\"   → 04_Demo_Prediction.ipynb (Demo) [2 phút]\\n\")\n    \n    print(\"📝 BƯỚC 3: Dự Đoán Cho Bệnh Nhân (2 phút)\")
    print(\"-\" * 70)
    print(\"Python code:\")\n    print("="*70)
    print("\nDocumentation:")
    print("  - README.md ................. Complete guide")
    print("  - GETTING_STARTED.md ....... Step-by-step guide")
    print("  - PROJECT_SUMMARY.md ....... Project summary")
    print("  - FILE_INDEX.md ............ Files list\n")\n\ndef main():\n    \"\"\"Hàm main\"\"\"\n    print_banner()\n    \n    # Kiểm tra files\n    files_ok = check_files()\n    \n    # Kiểm tra dữ liệu\n    data_ok = check_data()\n    \n    # Kiểm tra mô hình\n    models_ok = check_models()\n    \n    # In hướng dẫn nhanh\n    print_quick_start()\n    \n    # Tóm tắt trạng thái\n    print(\"📋 TRẠNG THÁI HỆỆN TẠI\")\n    print(\"=\"*70)\n    print(f\"Files cần thiết: {'✅ OK' if files_ok else '❌ Thiếu'}\")\n    print(f\"Dữ liệu: {'✅ OK' if data_ok else '⏳ Chưa tải'}\")\n    print(f\"Mô hình: {'✅ OK' if models_ok else '⏳ Chưa tạo'}\")\n    print(\"=\"*70 + \"\\n\")\n    \n    if files_ok and data_ok and models_ok:\n        print(\"🎉 Tất cả chuẩn bị xong! Sẵn sàng sử dụng!\\n\")\n    elif files_ok and data_ok:\n        print(\"⏳ Bước tiếp theo: Chạy notebooks để tạo mô hình\\n\")\n    else:\n        print(\"⏳ Bước tiếp theo: Tải dữ liệu và cài thư viện\\n\")\n\nif __name__ == \"__main__\":\n    try:\n        main()\n    except Exception as e:\n        print(f\"❌ Lỗi: {e}\")\n        sys.exit(1)
