"""
Demo Application
Ứng dụng demo dự đoán nguy cơ tiểu đường
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class DiabetesPredictionDemo:
    """Ứng dụng demo dự đoán bệnh tiểu đường"""
    
    # Risk levels
    RISK_LEVELS = {
        'low': (0.0, 0.4, 'Thấp - Ít nguy cơ mắc tiểu đường'),
        'medium': (0.4, 0.7, 'Trung bình - Có đấu hiệu mắc tiểu đường'),
        'high': (0.7, 1.0, 'Cao - Nguy cơ cao mắc tiểu đường')
    }
    
    def __init__(self, model, scaler, feature_names):
        """
        Khởi tạo demo
        model: mô hình đã huấn luyện
        scaler: StandardScaler đã fit trên dữ liệu huấn luyện
        feature_names: danh sách tên đặc trưng
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
    
    def predict_risk(self, patient_data):
        """
        Dự đoán nguy cơ tiểu đường cho bệnh nhân
        patient_data: dict hoặc list chứa 8 giá trị đầu vào
        Trả về: (probability, risk_level, risk_description)
        """
        # Chuyển thành numpy array
        if isinstance(patient_data, dict):
            data = np.array([patient_data[name] for name in self.feature_names]).reshape(1, -1)
        else:
            data = np.array(patient_data).reshape(1, -1)
        
        # Chuẩn hóa
        data_scaled = self.scaler.transform(data)
        
        # Dự đoán xác suất
        probability = self.model.predict_proba(data_scaled)[0, 1]
        
        # Phân loại mức rủi ro
        risk_level = self._get_risk_level(probability)
        
        return probability, risk_level
    
    def _get_risk_level(self, probability):
        """Phân loại mức rủi ro dựa trên xác suất"""
        for level, (min_prob, max_prob, description) in self.RISK_LEVELS.items():
            if min_prob <= probability < max_prob:
                return level, description
        
        # Nếu xác suất = 1.0
        return 'high', self.RISK_LEVELS['high'][2]
    
    def interactive_prediction(self):
        """Chế độ dự đoán tương tác (nhập dữ liệu từ keyboard)"""
        print("\n" + "="*60)
        print("🏥 HỆ THỐNG DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG")
        print("="*60)
        print("\nNhập thông tin bệnh nhân:")
        print("(Nhập 'exit' để thoát)\n")
        
        patient_data = {}
        
        feature_info = {
            'Pregnancies': 'Số lần mang thai (0-20)',
            'Glucose': 'Nồng độ glucose (mg/dL)',
            'BloodPressure': 'Huyết áp (mmHg)',
            'SkinThickness': 'Độ dày da (mm)',
            'Insulin': 'Nồng độ insulin (mIU/L)',
            'BMI': 'Chỉ số khối cơ thể',
            'DiabetesPedigreeFunction': 'Chỉ số di truyền (0-2.5)',
            'Age': 'Tuổi (năm)'
        }
        
        try:
            for feature in self.feature_names:
                while True:
                    print(f"{feature} ({feature_info.get(feature, 'N/A')}): ", end="")
                    user_input = input().strip()
                    
                    if user_input.lower() == 'exit':
                        return
                    
                    try:
                        value = float(user_input)
                        if value < 0:
                            print("⚠️  Giá trị phải không âm. Thử lại!")
                            continue
                        patient_data[feature] = value
                        break
                    except ValueError:
                        print("❌ Vui lòng nhập số hợp lệ!")
            
            # Dự đoán
            probability, (risk_level, description) = self.predict_risk(patient_data)
            
            # Hiển thị kết quả
            print("\n" + "="*60)
            print("📋 KẾT QUẢ DỰ ĐOÁN")
            print("="*60)
            
            for feature in self.feature_names:
                print(f"  {feature}: {patient_data[feature]}")
            
            print("\n" + "-"*60)
            print(f"Xác suất mắc tiểu đường: {probability:.2%}")
            print(f"Mức rủi ro: {description}")
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nThoát chương trình.")
    
    def predict_batch(self, patients_list):
        """Dự đoán cho nhóm bệnh nhân"""
        results = []
        
        for i, patient_data in enumerate(patients_list, 1):
            probability, (risk_level, description) = self.predict_risk(patient_data)
            results.append({
                'Patient': f'Patient {i}',
                'Probability': f'{probability:.2%}',
                'Risk Level': description
            })
        
        return results


def main():
    """Test demo"""
    print("Module Demo đã sẵn sàng!")
    print("Sử dụng: from src.demo import DiabetesPredictionDemo")


if __name__ == "__main__":
    main()
