"""
Model Training Module
Huấn luyện các mô hình: Logistic Regression, Random Forest, XGBoost
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score


class DiabetesModelTrainer:
    """Huấn luyện các mô hình dự đoán"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_logistic_regression(self, X_train, y_train):
        """Huấn luyện Logistic Regression"""
        print("\n🔹 Huấn luyện Logistic Regression...")
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"   ✓ Cross-validation F1-Score: {cv_score.mean():.4f} (+/- {cv_score.std():.4f})")
        
        self.models['Logistic Regression'] = model
        return model
    
    def train_random_forest(self, X_train, y_train, n_estimators=100):
        """Huấn luyện Random Forest"""
        print("\n🔹 Huấn luyện Random Forest...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"   ✓ Cross-validation F1-Score: {cv_score.mean():.4f} (+/- {cv_score.std():.4f})")
        
        self.models['Random Forest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, n_estimators=100):
        """Huấn luyện XGBoost"""
        print("\n🔹 Huấn luyện XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"   ✓ Cross-validation F1-Score: {cv_score.mean():.4f} (+/- {cv_score.std():.4f})")
        
        self.models['XGBoost'] = model
        return model
    
    def train_knn(self, X_train, y_train, n_neighbors=5):
        """Huấn luyện KNN"""
        print("\n🔹 Huấn luyện KNN...")
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"   ✓ Cross-validation F1-Score: {cv_score.mean():.4f} (+/- {cv_score.std():.4f})")
        
        self.models['KNN'] = model
        return model
    
    def get_model(self, model_name):
        """Lấy mô hình theo tên"""
        return self.models.get(model_name)
    
    def get_all_models(self):
        """Lấy tất cả mô hình đã huấn luyện"""
        return self.models
    
    def set_best_model(self, model_name):
        """Đặt mô hình tốt nhất"""
        if model_name in self.models:
            self.best_model = self.models[model_name]
            self.best_model_name = model_name
            print(f"\n✓ Mô hình tốt nhất được chọn: {model_name}")
        else:
            print(f"❌ Mô hình '{model_name}' không tồn tại!")


def main():
    """Test model training"""
    print("Module Models Training đã sẵn sàng!")
    print("Sử dụng: from src.models import DiabetesModelTrainer")


if __name__ == "__main__":
    main()
