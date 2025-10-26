#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download Pima Indians Diabetes Dataset from Kaggle
"""

import os
import pandas as pd
import urllib.request
import ssl

# Disable SSL verification for download
ssl._create_default_https_context = ssl._create_unverified_context

print("📥 Đang tải Pima Indians Diabetes Dataset từ Kaggle...\n")

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
output_file = "data/diabetes_full.csv"

try:
    print(f"URL: {url}")
    print(f"Tệp đầu ra: {output_file}\n")
    
    # Download the file
    urllib.request.urlretrieve(url, output_file)
    
    # Load and check
    df = pd.read_csv(output_file, header=None)
    
    # Add column names
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df.columns = columns
    
    # Save with headers
    df.to_csv(output_file, index=False)
    
    print(f"✅ Tải thành công!")
    print(f"   Dataset: {df.shape[0]} mẫu × {df.shape[1]} cột")
    print(f"\n   Outcome Distribution:")
    print(f"   Class 0 (No Diabetes): {(df['Outcome']==0).sum()}")
    print(f"   Class 1 (Diabetes):    {(df['Outcome']==1).sum()}")
    
    # Replace main file
    import shutil
    shutil.copy(output_file, "data/diabetes.csv")
    print(f"\n✓ Cập nhật: data/diabetes.csv")
    
except Exception as e:
    print(f"❌ Lỗi: {e}")
    print(f"Vui lòng tải thủ công từ:")
    print(f"https://www.kaggle.com/uciml/pima-indians-diabetes-database")
