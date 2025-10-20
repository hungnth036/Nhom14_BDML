"""
Model Evaluation Module
Đánh giá mô hình: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)


class DiabetesModelEvaluator:
    """Đánh giá hiệu suất mô hình"""
    
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Đánh giá mô hình trên tập test
        Trả về từ điển các metrics
        """
        # Dự đoán
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Tính toán metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        }
        
        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # In kết quả
        print(f"\n📊 Kết quả đánh giá: {model_name}")
        print("=" * 50)
        print(f"Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall:    {metrics['Recall']:.4f}")
        print(f"F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        
        return metrics
    
    def evaluate_multiple_models(self, models_dict, X_test, y_test):
        """
        Đánh giá nhiều mô hình cùng lúc
        models_dict: dict {model_name: model_object}
        """
        print("\n" + "="*60)
        print("🔍 ĐÁNH GIÁ NHIỀU MÔ HÌNH")
        print("="*60)
        
        results_list = []
        for model_name, model in models_dict.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results_list.append(metrics)
        
        # Tạo bảng so sánh
        results_df = pd.DataFrame(results_list)
        
        print("\n📋 So sánh các mô hình:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_confusion_matrix(self, model_name, y_test, y_pred, save_path=None):
        """Vẽ Confusion Matrix"""
        if model_name not in self.results:
            print(f"❌ Mô hình '{model_name}' chưa được đánh giá!")
            return
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Thực tế (Actual)')
        plt.xlabel('Dự đoán (Predicted)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Lưu: {save_path}")
        
        plt.tight_layout()
        return plt
    
    def plot_feature_importance(self, model, model_name, feature_names=None, save_path=None):
        """Vẽ Feature Importance (cho Random Forest, XGBoost)"""
        if not hasattr(model, 'feature_importances_'):
            print(f"❌ Mô hình '{model_name}' không hỗ trợ feature importance!")
            return
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Sắp xếp
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Lưu: {save_path}")
        
        return plt
    
    def print_classification_report(self, model_name):
        """In chi tiết Classification Report"""
        if model_name not in self.results:
            print(f"❌ Mô hình '{model_name}' chưa được đánh giá!")
            return
        
        metrics = self.results[model_name]['metrics']
        print(f"\n📄 Classification Report: {model_name}")
        print("="*50)
        print(f"True Negatives:  {self.results[model_name]['confusion_matrix'][0, 0]}")
        print(f"False Positives: {self.results[model_name]['confusion_matrix'][0, 1]}")
        print(f"False Negatives: {self.results[model_name]['confusion_matrix'][1, 0]}")
        print(f"True Positives:  {self.results[model_name]['confusion_matrix'][1, 1]}")
    
    def get_results_dataframe(self):
        """Lấy DataFrame của tất cả kết quả"""
        metrics_list = [self.results[name]['metrics'] for name in self.results.keys()]
        return pd.DataFrame(metrics_list)


def main():
    """Test evaluation"""
    print("Module Evaluation đã sẵn sàng!")
    print("Sử dụng: from src.evaluation import DiabetesModelEvaluator")


if __name__ == "__main__":
    main()
