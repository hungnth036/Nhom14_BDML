#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create enhanced dataset with feature engineering
"""

import pandas as pd
import numpy as np
import os

print("\n📊 Tạo dataset cải tiến...")
print("-" * 70)

# Đọc dữ liệu hiện tại
df = pd.read_csv('data/diabetes.csv')

print(f"Dataset hiện tại: {df.shape[0]} mẫu")

# Kiểm tra xem có bao nhiêu mẫu
if df.shape[0] < 200:
    print(f"Dataset quá nhỏ ({df.shape[0]} mẫu), tạo dữ liệu mở rộng...")
    
    # Lấy mẫu hiện tại và tạo biến thể
    original_df = df.copy()
    
    # Thêm Gaussian noise để tạo mẫu mới (data augmentation)
    np.random.seed(42)
    
    augmented_dfs = [original_df]
    
    for i in range(5):  # Tạo 5 lần lặp lại
        augmented = original_df.copy()
        
        # Thêm noise nhỏ vào các features (ngoại trừ Outcome)
        for col in augmented.columns[:-1]:
            noise = np.random.normal(0, augmented[col].std() * 0.05, len(augmented))
            augmented[col] = augmented[col] + noise
            
            # Đảm bảo các giá trị hợp lệ
            if col in ['Pregnancies', 'Age']:
                augmented[col] = augmented[col].astype(int).clip(0, None)
            else:
                augmented[col] = augmented[col].clip(0, None)
        
        augmented_dfs.append(augmented)
    
    # Gộp tất cả
    df_expanded = pd.concat(augmented_dfs, ignore_index=True)
    
    # Lưu
    df_expanded.to_csv('data/diabetes_expanded.csv', index=False)
    df_expanded.to_csv('data/diabetes.csv', index=False)
    
    print(f"✓ Dataset mở rộng: {df_expanded.shape[0]} mẫu")
    print(f"  Outcome: {(df_expanded['Outcome']==0).sum()} negative, {(df_expanded['Outcome']==1).sum()} positive")

else:
    print(f"✓ Dataset đủ lớn: {df.shape[0]} mẫu")

print("\n" + "-" * 70 + "\n")
