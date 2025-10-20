"""
Data Preprocessing Module
Xử lý dữ liệu: xóa giá trị thiếu, chuẩn hóa dữ liệu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DiabetesDataPreprocessor:
    """Xử lý dữ liệu bệnh tiểu đường"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filepath):
        """Tải dữ liệu từ CSV"""
        df = pd.read_csv(filepath)
        print(f"✓ Tải dữ liệu thành công: {df.shape[0]} mẫu, {df.shape[1]} đặc trưng")
        return df
    
    def handle_missing_values(self, df):
        """
        Xử lý giá trị thiếu/không hợp lệ
        Các cột như Glucose, BMI, BloodPressure không nên bằng 0
        """
        df_clean = df.copy()
        
        # Danh sách cột không nên có giá trị 0
        cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        print("\n📊 Xử lý giá trị không hợp lệ (= 0):")
        for col in cols_with_zero:
            zero_count = (df_clean[col] == 0).sum()
            if zero_count > 0:
                # Thay bằng median
                median_val = df_clean[df_clean[col] != 0][col].median()
                df_clean.loc[df_clean[col] == 0, col] = median_val
                print(f"  - {col}: thay {zero_count} giá trị 0 bằng median ({median_val:.2f})")
        
        return df_clean
    
    def normalize_features(self, X_train, X_test=None):
        """Chuẩn hóa dữ liệu"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_data(self, df, test_size=0.2):
        """
        Chuẩn bị dữ liệu: tách input/output, split train/test, chuẩn hóa
        """
        # Tách X, y
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        self.feature_names = X.columns.tolist()
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\n✓ Tách train/test:")
        print(f"  - Train: {X_train.shape[0]} mẫu")
        print(f"  - Test: {X_test.shape[0]} mẫu")
        print(f"  - Train - Outcome=0: {(y_train==0).sum()}, Outcome=1: {(y_train==1).sum()}")
        print(f"  - Test  - Outcome=0: {(y_test==0).sum()}, Outcome=1: {(y_test==1).sum()}")
        
        # Chuẩn hóa
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)
        
        print(f"✓ Chuẩn hóa dữ liệu hoàn tất")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_feature_names(self):
        """Lấy tên các đặc trưng"""
        return self.feature_names


def main():
    """Test preprocessing"""
    # Download dữ liệu từ Kaggle hoặc sử dụng dữ liệu có sẵn
    preprocessor = DiabetesDataPreprocessor()
    
    # Ví dụ
    print("Module Preprocessing đã sẵn sàng!")
    print("Sử dụng: from src.preprocessing import DiabetesDataPreprocessor")


if __name__ == "__main__":
    main()
