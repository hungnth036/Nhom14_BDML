# 🚀 Hướng Dẫn Bắt Đầu (Getting Started)

## Bước 1: Chuẩn Bị Dữ Liệu

### 1.1 Tải Dữ Liệu từ Kaggle

1. Truy cập: https://www.kaggle.com/uciml/pima-indians-diabetes-database
2. Click nút "Download" để tải file `diabetes.csv`
3. Giải nén file và đặt vào thư mục: `final/data/diabetes.csv`

### 1.2 Kiểm Tra Dữ Liệu
```bash
# Kiểm tra xem file đã ở đúng vị trí
ls final/data/diabetes.csv
# Hoặc trên Windows
dir final\data\diabetes.csv
```

---

## Bước 2: Cài Đặt Thư Viện

### 2.1 Cài Đặt Dependencies
```bash
# Đi vào thư mục dự án
cd final

# Cài tất cả thư viện cần thiết
pip install -r requirements.txt
```

### 2.2 Kiểm Tra Cài Đặt
```bash
# Kiểm tra Python version
python --version  # Phải ≥ 3.7

# Kiểm tra các thư viện chính
python -c "import pandas, sklearn, xgboost, jupyter; print('✓ All libraries installed!')"
```

---

## Bước 3: Chạy Phân Tích (EDA)

### 3.1 Khởi Động Jupyter Notebook
```bash
# Từ thư mục dự án
jupyter notebook

# Trình duyệt sẽ mở tự động
# Nếu không, truy cập: http://localhost:8888
```

### 3.2 Chạy Notebook EDA
1. Mở file: `notebooks/01_EDA.ipynb`
2. Nhấn `Cell` → `Run All` hoặc `Ctrl+Enter` để chạy từng cell
3. Xem các biểu đồ và thống kê về dữ liệu

**Kết quả expected:**
- Biểu đồ phân phối các đặc trưng
- Ma trận tương quan
- Phân tích giá trị 0 không hợp lệ

---

## Bước 4: Huấn Luyện Mô Hình

### 4.1 Chạy Notebook Model Training
1. Mở file: `notebooks/02_Model_Training.ipynb`
2. Chạy tất cả cells
3. Chương trình sẽ:
   - Xử lý dữ liệu
   - Chuẩn hóa đặc trưng
   - Huấn luyện 4 mô hình
   - Lưu các mô hình vào thư mục `models/`

**Thời gian:** ~5-10 phút

**Kết quả expected:**
- 4 file `.pkl` được lưu trong `models/`
- Biểu đồ so sánh Cross-Validation Scores

---

## Bước 5: Đánh Giá Mô Hình

### 5.1 Chạy Notebook Evaluation
1. Mở file: `notebooks/03_Model_Evaluation.ipynb`
2. Chạy tất cả cells
3. Chương trình sẽ:
   - Tải mô hình từ thư mục `models/`
   - Tính toán các metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
   - Vẽ biểu đồ so sánh

**Kết quả expected:**
- Bảng so sánh 4 mô hình
- Confusion Matrix cho mỗi mô hình
- ROC Curve
- Feature Importance ranking

---

## Bước 6: Dự Đoán Cho Bệnh Nhân

### 6.1 Chạy Notebook Demo Prediction
1. Mở file: `notebooks/04_Demo_Prediction.ipynb`
2. Chạy tất cả cells
3. Chương trình sẽ:
   - Dự đoán cho 3 ví dụ bệnh nhân
   - So sánh dự đoán giữa 4 mô hình
   - Đưa ra khuyến nghị y tế

**Ví dụ Kết Quả:**
```
🏥 KẾT QUẢ DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG
======================================================================
📋 Thông tin bệnh nhân:
  Pregnancies:........................ 6
  Glucose:............................ 175
  BloodPressure:..................... 72
  SkinThickness:..................... 35
  Insulin:........................... 148
  BMI:............................... 38.5
  DiabetesPedigreeFunction:.......... 0.605
  Age:.............................. 48

🔍 Kết quả dự đoán:
  Xác suất mắc tiểu đường: 82.45%
  Mức rủi ro: Cao
  🔴 Nguy cơ cao mắc tiểu đường

💡 KHUYẾN NGỊ VÀ HÀNH ĐỘNG
======================================================================
🔴 NGUY CƠ CAO:
- CẦN KIỂM TRA VÀ THĂM KHÁ BÁC SĨ NGAY
- Làm xét nghiệm glucose, HbA1c
- Có thể cần vào chế độ điều trị
- Giảm cân nhanh chóng
- Bắt đầu chương trình tập luyện
- Kiểm tra hàng 3 tháng
```

---

## Bước 7: Sử Dụng Mô Hình Cho Dự Đoán Mới

### 7.1 Sử Dụng Python Script
```python
import sys
sys.path.append('path/to/final')

from src.demo import DiabetesPredictionDemo
import joblib

# Tải mô hình
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Khởi tạo
demo = DiabetesPredictionDemo(model, scaler, feature_names)

# Dự đoán
patient = {
    'Pregnancies': 2,
    'Glucose': 120,
    'BloodPressure': 70,
    'SkinThickness': 24,
    'Insulin': 110,
    'BMI': 32.0,
    'DiabetesPedigreeFunction': 0.35,
    'Age': 42
}

prob, (risk_level, desc) = demo.predict_risk(patient)
print(f"Xác suất: {prob:.2%}")
print(f"Mức rủi ro: {desc}")
```

### 7.2 Sử Dụng Jupyter Notebook
- Sử dụng các cell trong `04_Demo_Prediction.ipynb`
- Thay đổi giá trị `patient_data` để dự đoán cho bệnh nhân khác

---

## Kiểm Tra & Xác Minh

### ✅ Checklist Hoàn Thành
- [ ] Đã tải file `diabetes.csv`
- [ ] Đã cài đặt tất cả thư viện
- [ ] Chạy thành công `01_EDA.ipynb`
- [ ] Chạy thành công `02_Model_Training.ipynb`
- [ ] Chạy thành công `03_Model_Evaluation.ipynb`
- [ ] Chạy thành công `04_Demo_Prediction.ipynb`
- [ ] Có thể dự đoán cho bệnh nhân mới

---

## Gỡ Lỗi (Troubleshooting)

### Lỗi 1: "No module named 'pandas'"
```bash
# Giải pháp
pip install pandas numpy scikit-learn xgboost
```

### Lỗi 2: "FileNotFoundError: diabetes.csv not found"
```bash
# Kiểm tra vị trí file
# Phải ở: final/data/diabetes.csv

# Hoặc chỉnh sửa đường dẫn trong notebook
data_path = 'path/to/your/diabetes.csv'
```

### Lỗi 3: "Jupyter command not found"
```bash
# Cài Jupyter
pip install jupyter

# Hoặc sử dụng JupyterLab
pip install jupyterlab
jupyter lab
```

### Lỗi 4: Models không được tìm thấy
```bash
# Kiểm tra thư mục models tồn tại
# final/models/

# Nếu không, chạy lại notebook 2 để tạo mô hình
```

---

## Tips & Tricks

### 💡 Mẹo 1: Chạy Notebook Nhanh Hơn
```python
# Thêm vào đầu notebook
%matplotlib inline
%load_ext autoreload
%autoreload 2
```

### 💡 Mẹo 2: Lưu Biểu Đồ
```python
plt.savefig('results/my_plot.png', dpi=300, bbox_inches='tight')
```

### 💡 Mẹo 3: Tăng Tốc XGBoost
```python
xgb_model = xgb.XGBClassifier(
    n_jobs=-1,  # Sử dụng tất cả CPU cores
    tree_method='gpu_hist'  # Sử dụng GPU (nếu có)
)
```

### 💡 Mẹo 4: Batch Prediction
```python
# Dự đoán cho nhiều bệnh nhân
patients = [patient_1, patient_2, patient_3]
results = demo.predict_batch(patients)
```

---

## 📞 Hỗ Trợ & Hỏi Đáp

### Q1: Làm thế nào để cập nhật dữ liệu?
**A:** Tải dữ liệu mới từ Kaggle, thay thế file `diabetes.csv`, rồi chạy lại các notebook.

### Q2: Có thể thêm các mô hình khác không?
**A:** Có! Thêm code trong `02_Model_Training.ipynb` và `03_Model_Evaluation.ipynb`.

### Q3: Dự đoán có chính xác không?
**A:** Mô hình có Accuracy ~78%, nhưng chỉ là hỗ trợ, không thay thế chẩn đoán bác sĩ!

### Q4: Có thể deploy mô hình lên web không?
**A:** Có! Sử dụng Flask, Django, hoặc Streamlit để tạo web interface.

---

## 📚 Tài Liệu Bổ Sung

- [Machine Learning Basics](https://scikit-learn.org/stable/modules/classification.html)
- [Feature Scaling](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Kaggle Notebooks](https://www.kaggle.com/notebooks)

---

**Chúc bạn thành công! 🎉**

Mọi thắc mắc, vui lòng tham khảo file `README.md` hoặc các notebook có các comment chi tiết.
