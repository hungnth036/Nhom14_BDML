#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download Pima Indians Diabetes Dataset from alternative source
"""

import pandas as pd
import os
from pathlib import Path

print("\n📥 Downloading Pima Indians Diabetes Dataset...")
print("-" * 70)

# Try to load from online source
try:
    # Alternative URL - using direct data source
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data"
    
    print(f"Fetching from: {url}")
    
    df = pd.read_csv(url, header=None)
    
    # Add column names
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    df.columns = columns
    
    # Save to data folder
    os.makedirs('data', exist_ok=True)
    output_path = 'data/diabetes.csv'
    
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Success!")
    print(f"   Dataset: {df.shape[0]} samples × {df.shape[1]} features")
    print(f"   Saved to: {output_path}")
    print(f"\n   Outcome distribution:")
    print(f"   - No diabetes (0): {(df['Outcome']==0).sum()} samples")
    print(f"   - Diabetes (1): {(df['Outcome']==1).sum()} samples")
    
except Exception as e:
    print(f"\n❌ Failed to download: {e}")
    print("\nAlternative: Manual download from Kaggle")
    print("1. Visit: https://www.kaggle.com/uciml/pima-indians-diabetes-database")
    print("2. Download diabetes.csv")
    print("3. Place in: data/diabetes.csv")

print("\n" + "-" * 70 + "\n")
