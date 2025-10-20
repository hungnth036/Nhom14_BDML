# 🏥 Dự Đoán Bệnh Tiểu Đường từ Dữ Liệu Y Tế

## 📋 Mô Tả Dự Án

Đây là một dự án **Machine Learning** nhằm dự đoán khả năng mắc bệnh tiểu đường (Diabetes) dựa trên dữ liệu sức khỏe của bệnh nhân, sử dụng **Pima Indians Diabetes Database** từ Kaggle.

### 🎯 Mục Tiêu
- Phân tích và khám phá dữ liệu y tế thực tế
- Xây dựng các mô hình Machine Learning để dự đoán bệnh tiểu đường
- Đánh giá hiệu suất các mô hình bằng nhiều metrics khác nhau
- Phân loại mức rủi ro (Thấp, Trung bình, Cao)
- Cung cấp công cụ dự đoán hỗ trợ sàng lọc bệnh nhân

---

## 📁 Cấu Trúc Dự Án

```
final/
├── data/                          # Thư mục chứa dữ liệu
│   └── diabetes.csv               # File dữ liệu từ Kaggle (cần tải)
│
├── notebooks/                     # Jupyter Notebooks
│   ├── 01_EDA.ipynb               # Phân tích khám phá dữ liệu
│   ├── 02_Model_Training.ipynb    # Huấn luyện các mô hình
│   ├── 03_Model_Evaluation.ipynb  # Đánh giá mô hình
│   └── 04_Demo_Prediction.ipynb   # Demo ứng dụng dự đoán
│
├── src/                           # Mã nguồn Python
│   ├── __init__.py
│   ├── preprocessing.py           # Xử lý dữ liệu
│   ├── models.py                  # Huấn luyện mô hình
│   ├── evaluation.py              # Đánh giá mô hình
│   └── demo.py                    # Demo ứng dụng
│
├── models/                        # Thư mục chứa mô hình đã huấn luyện
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── results/                       # Thư mục kết quả
│   └── (chứa các biểu đồ và kết quả)
│
├── requirements.txt               # Danh sách thư viện cần thiết
└── README.md                      # File hướng dẫn (file này)
```

---

## 📊 Thông Tin Dữ Liệu

### Nguồn Dữ Liệu
- **Pima Indians Diabetes Database** từ Kaggle
- URL: https://www.kaggle.com/uciml/pima-indians-diabetes-database

### Kích Thước Dữ Liệu
- **768 quan sát** (bệnh nhân nữ trên 21 tuổi)
- **9 cột** (8 đặc trưng + 1 biến mục tiêu)

### Các Đặc Trưng (Features)

| Đặc Trưng | Mô Tả | Kiểu Dữ Liệu |
|-----------|-------|-------------|
| Pregnancies | Số lần mang thai | INT |
| Glucose | Nồng độ glucose trong máu (mg/dL) | FLOAT |
| BloodPressure | Huyết áp (mmHg) | FLOAT |
| SkinThickness | Độ dày da (mm) | FLOAT |
| Insulin | Nồng độ insulin (mIU/L) | FLOAT |
| BMI | Chỉ số khối cơ thể | FLOAT |
| DiabetesPedigreeFunction | Chỉ số di truyền liên quan tiểu đường | FLOAT |
| Age | Tuổi (năm) | INT |
| Outcome | **[Target]** 0 = Không, 1 = Có tiểu đường | INT |

---

## 🛠️ Công Nghệ Sử Dụng

### Ngôn Ngữ & Framework
- **Python 3.x**
- **Jupyter Notebook** (cho phân tích tương tác)
- **Apache Spark** (tùy chọn, cho Big Data)

### Thư Viện Chính
- **Pandas, NumPy** - Xử lý dữ liệu
- **Scikit-learn** - Các mô hình ML cơ bản
- **XGBoost** - Gradient Boosting
- **Matplotlib, Seaborn** - Trực quan hóa dữ liệu
- **Joblib** - Lưu/tải mô hình

---

## 📚 Các Mô Hình Machine Learning

Dự án huấn luyện và so sánh **4 mô hình**:

| Mô Hình | Loại | Ưu Điểm |
|---------|------|--------|
| **Logistic Regression** | Phân loại (Classification) | Đơn giản, dễ diễn giải |
| **Random Forest** | Ensemble (Cây quyết định) | Hiệu suất cao, xem được Feature Importance |
| **XGBoost** | Gradient Boosting | Mạnh mẽ, cắt được Overfitting |
| **KNN** | Lazy Learning | Nhanh khi có dữ liệu nhỏ |

---

## 📈 Các Metrics Đánh Giá

### Độ Đo Chính
- **Accuracy** - Tỷ lệ dự đoán chính xác tổng thể
- **Precision** - Độ chính xác khi dự đoán "mắc bệnh"
- **Recall (Sensitivity)** - Khả năng phát hiện đúng bệnh nhân mắc bệnh
- **F1-Score** - Cân bằng giữa Precision và Recall
- **ROC-AUC** - Diện tích dưới đường cong ROC

### Mục Tiêu Kỳ Vọng
- ✅ Accuracy ≥ 75%
- ✅ F1-Score ≥ 0.70
- ✅ ROC-AUC ≥ 0.80

---

## 🚀 Hướng Dẫn Sử Dụng

### 1️⃣ Chuẩn Bị Môi Trường

#### A. Tải Dữ Liệu
```bash
# Tải file diabetes.csv từ Kaggle:
# https://www.kaggle.com/uciml/pima-indians-diabetes-database
# Đặt vào thư mục: final/data/diabetes.csv
```

#### B. Cài Đặt Thư Viện
```bash
# Sử dụng pip
pip install -r requirements.txt

# Hoặc cài từng gói (nếu cần)
pip install pandas numpy scikit-learn xgboost jupyter matplotlib seaborn
```

### 2️⃣ Chạy Phân Tích

#### Phương Pháp 1: Sử dụng Jupyter Notebook (Khuyến Nghị)
```bash
# Chạy Jupyter
jupyter notebook

# Mở từng notebook theo thứ tự:
# 1. 01_EDA.ipynb - Khám phá dữ liệu
# 2. 02_Model_Training.ipynb - Huấn luyện mô hình
# 3. 03_Model_Evaluation.ipynb - Đánh giá mô hình
# 4. 04_Demo_Prediction.ipynb - Demo dự đoán
```

#### Phương Pháp 2: Sử dụng Python Scripts
```bash
# Chạy từng module
python src/preprocessing.py
python src/models.py
python src/evaluation.py
python src/demo.py
```

### 3️⃣ Dự Đoán Cho Bệnh Nhân Mới

```python
from src.demo import DiabetesPredictionDemo
import joblib

# Tải mô hình
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Khởi tạo demo
demo = DiabetesPredictionDemo(model, scaler, feature_names)

# Dự đoán cho bệnh nhân
patient_data = {
    'Pregnancies': 6,
    'Glucose': 175,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 148,
    'BMI': 38.5,
    'DiabetesPedigreeFunction': 0.605,
    'Age': 48
}

probability, (risk_level, description) = demo.predict_risk(patient_data)
print(f"Xác suất: {probability:.2%}")
print(f"Mức rủi ro: {description}")
```

---

## 📊 Kết Quả Kỳ Vọng

### Hiệu Suất Mô Hình
- **Random Forest** thường cho kết quả tốt nhất
- Tất cả mô hình đều đạt Accuracy > 75%
- ROC-AUC > 0.84 cho các mô hình tốt

### Các Yếu Tố Ảnh Hưởng Mạnh Nhất
1. **Glucose** - Nồng độ glucose
2. **BMI** - Chỉ số khối cơ thể
3. **Age** - Tuổi
4. **Insulin** - Nồng độ insulin

### Phân Loại Rủi Ro
- 🟢 **Thấp** (P < 0.4): Ít nguy cơ
- 🟡 **Trung bình** (0.4 ≤ P ≤ 0.7): Có đấu hiệu
- 🔴 **Cao** (P ≥ 0.7): Nguy cơ cao

---

## 💡 Khuyến Nghị Dựa Trên Mức Rủi Ro

### Nguy Cơ Thấp (P < 0.4)
- ✅ Duy trì lối sống lành mạnh
- ✅ Tập thể dục 150 phút/tuần
- ✅ Ăn uống cân bằng, tránh đường
- ✅ Kiểm tra định kỳ 2 năm/lần

### Nguy Cơ Trung Bình (0.4 ≤ P ≤ 0.7)
- ⚠️ Tăng cường kiểm soát
- ⚠️ Giảm cân nếu BMI cao
- ⚠️ Tập thể dục 30 phút hàng ngày
- ⚠️ Kiểm tra hàng năm
- ⚠️ Tham khảo bác sĩ về chế độ ăn

### Nguy Cơ Cao (P ≥ 0.7)
- 🔴 **KIỂM TRA NGAY VỚI BÁC SĨ**
- 🔴 Làm xét nghiệm glucose, HbA1c
- 🔴 Có thể cần vào chế độ điều trị
- 🔴 Giảm cân nhanh chóng
- 🔴 Bắt đầu chương trình tập luyện
- 🔴 Kiểm tra hàng 3 tháng

---

## 🔧 Troubleshooting

### Lỗi: "File diabetes.csv not found"
```bash
# Giải pháp: Tải dữ liệu từ Kaggle
# https://www.kaggle.com/uciml/pima-indians-diabetes-database
# Đặt file vào: final/data/diabetes.csv
```

### Lỗi: "Module not found"
```bash
# Giải pháp: Cài đặt lại thư viện
pip install -r requirements.txt --upgrade
```

### Lỗi: Jupyter not found
```bash
# Giải pháp: Cài Jupyter
pip install jupyter
```

---

## 📚 Tài Liệu Tham Khảo

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Kaggle Pima Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

---

## 👥 Thành Viên Nhóm

- Nguyễn Trọng Hưng
- Nguyễn Dương Thế Cường

---

## 📝 Lịch Trình Thực Hiện

| Ngày | Công Việc | Người Thực Hiện |
|------|----------|------------------|
| 22/09/2025 | Giới thiệu dự án & Tìm kiếm dữ liệu | Cả nhóm |
| 15/10/2025 | Phân tích EDA & Tiền xử lý dữ liệu | Cả nhóm |
| 18/10/2025 | Xây dựng mô hình & Demo | Cả nhóm |
| 18/10/2025 | Chuẩn bị báo cáo | Cả nhóm |

---

## 📄 Giấy Phép

Dự án này được tạo cho mục đích học tập và nghiên cứu.

---

## 💬 Liên Hệ & Hỗ Trợ

Nếu có bất kỳ câu hỏi hoặc gặp vấn đề, vui lòng tham khảo tài liệu hoặc liên hệ nhóm phát triển.

---

**Cập nhật lần cuối: 20/10/2025**
