# 📑 Danh Sách File Dự Án (File Index)

**Cập nhật:** 20/10/2025  
**Tổng số files:** 20+  
**Tổng dung lượng:** ~2 MB (chưa tính dữ liệu)

---

## 📋 Danh Sách Chi Tiết

### 🔝 Root Directory (Thư Mục Gốc)

| File | Kích Thước | Mô Tả |
|------|-----------|-------|
| `README.md` | ~12 KB | 📘 Hướng dẫn chính (Tiếng Việt) |
| `GETTING_STARTED.md` | ~8 KB | 🚀 Hướng dẫn bắt đầu nhanh |
| `PROJECT_SUMMARY.md` | ~10 KB | 📊 Tóm tắt dự án |
| `requirements.txt` | ~0.3 KB | 📦 Danh sách thư viện |
| `FILE_INDEX.md` | ~6 KB | 📑 File này - Danh sách files |

---

### 📁 Thư mục `src/` - Mã Nguồn Python

| File | Dòng Code | Mô Tả |
|------|-----------|-------|
| `src/__init__.py` | 12 | 🔧 Module initialization |
| `src/preprocessing.py` | 140+ | ⚙️ Xử lý & chuẩn hóa dữ liệu |
| `src/models.py` | 130+ | 🤖 Huấn luyện 4 mô hình ML |
| `src/evaluation.py` | 160+ | 📊 Đánh giá & visualize metrics |
| `src/demo.py` | 120+ | 🎯 Demo dự đoán + khuyến nghị |

**Tổng:** 5 files, ~560+ dòng code

---

### 📓 Thư mục `notebooks/` - Jupyter Notebooks

| File | Cell | Dòng Code | Mô Tả |
|------|------|-----------|-------|
| `01_EDA.ipynb` | 20+ | 400+ | 🔍 Exploratory Data Analysis |
| `02_Model_Training.ipynb` | 15+ | 350+ | 🤖 Huấn luyện & lưu mô hình |
| `03_Model_Evaluation.ipynb` | 18+ | 380+ | 📊 Đánh giá & biểu đồ so sánh |
| `04_Demo_Prediction.ipynb` | 16+ | 320+ | 🏥 Demo dự đoán + khuyến nghị |

**Tổng:** 4 notebooks, ~70 cells, ~1,450+ dòng code

---

### 📦 Thư mục `models/` - Mô Hình Đã Huấn Luyện

| File | Định Dạng | Kích Thước | Mô Tả |
|------|-----------|-----------|-------|
| `logistic_regression_model.pkl` | Pickle | ~3 KB | 📊 Mô hình Logistic Regression |
| `random_forest_model.pkl` | Pickle | ~180 KB | 🌳 Mô hình Random Forest |
| `xgboost_model.pkl` | Pickle | ~85 KB | ⚡ Mô hình XGBoost |
| `knn_model.pkl` | Pickle | ~2 KB | 📍 Mô hình KNN |
| `scaler.pkl` | Pickle | ~0.5 KB | 📐 StandardScaler (chuẩn hóa) |
| `feature_names.pkl` | Pickle | ~0.2 KB | 📋 Tên các đặc trưng |

**Tổng:** 6 files (tạo sau khi chạy 02_Model_Training.ipynb)

---

### 📁 Thư mục `data/`

| File | Kích Thước | Mô Tả |
|------|-----------|-------|
| `diabetes.csv` | ~24 KB | 📊 Pima Indians Diabetes Database |
| *(Tạm thời trống)* | - | ⏳ Cần tải từ Kaggle |

**Ghi chú:** File này **chưa có**, cần tải từ Kaggle:
- URL: https://www.kaggle.com/uciml/pima-indians-diabetes-database

---

### 📁 Thư mục `results/`

| File | Loại | Mô Tả |
|------|------|-------|
| *(Chưa tạo)* | - | 📈 Chứa biểu đồ & kết quả phân tích |

**Ghi chú:** Được tạo sau khi chạy các notebooks

---

## 📊 Thống Kê Tổng Hợp

```
📂 Cấu trúc dự án:
├── 📄 5 file .md (hướng dẫn)
├── 📦 1 file requirements.txt
├── 🐍 5 file .py (src/)
├── 📓 4 file .ipynb (notebooks/)
├── 📊 6 file .pkl (models/) - tạo sau
├── 📂 4 thư mục chính
└── ...

📈 Thống kê code:
- Python code: ~560+ dòng (src/)
- Jupyter code: ~1,450+ dòng (notebooks/)
- Tổng cộng: ~2,000+ dòng code
- Tổng file: 20+ files
- Dung lượng: ~2-3 MB (chưa tính dữ liệu)
```

---

## 🎯 Thứ Tự Chạy Notebooks

### 1️⃣ **01_EDA.ipynb** - Phân Tích Khám Phá Dữ Liệu
- ⏱️ Thời gian: ~5 phút
- 📥 Input: `data/diabetes.csv`
- 📤 Output: Biểu đồ và thống kê
- 🎯 Mục đích: Hiểu dữ liệu

### 2️⃣ **02_Model_Training.ipynb** - Huấn Luyện Mô Hình
- ⏱️ Thời gian: ~5-10 phút
- 📥 Input: `data/diabetes.csv`
- 📤 Output: 6 file `.pkl` trong `models/`
- 🎯 Mục đích: Tạo & lưu mô hình

### 3️⃣ **03_Model_Evaluation.ipynb** - Đánh Giá Mô Hình
- ⏱️ Thời gian: ~3-5 phút
- 📥 Input: `models/*.pkl`
- 📤 Output: Biểu đồ metrics, ROC, Feature Importance
- 🎯 Mục đích: So sánh hiệu suất

### 4️⃣ **04_Demo_Prediction.ipynb** - Demo Dự Đoán
- ⏱️ Thời gian: ~2-3 phút
- 📥 Input: `models/*.pkl`
- 📤 Output: Dự đoán & khuyến nghị
- 🎯 Mục đích: Thử nghiệm hệ thống

---

## 🔗 Quan Hệ Giữa Các Files

```
data/diabetes.csv
    ↓
01_EDA.ipynb (phân tích)
    ↓
02_Model_Training.ipynb (tạo models/*.pkl)
    ↓
├─→ 03_Model_Evaluation.ipynb (đánh giá)
│       ↓
└─→ 04_Demo_Prediction.ipynb (dự đoán)
        ↓
    results/ (biểu đồ & kết quả)

src/preprocessing.py ─→ (module dùng cho notebooks)
src/models.py ──────→ (module dùng cho notebooks)
src/evaluation.py ──→ (module dùng cho notebooks)
src/demo.py ────────→ (module dùng cho notebooks)
```

---

## 📝 Mô Tả Từng Loại File

### `.py` (Python Scripts)
- **Tác dụng:** Mã nguồn có thể tái sử dụng
- **Vị trí:** `src/`
- **Cách sử dụng:** Import vào notebook hoặc script khác
- **Ví dụ:**
  ```python
  from src.preprocessing import DiabetesDataPreprocessor
  preprocessor = DiabetesDataPreprocessor()
  ```

### `.ipynb` (Jupyter Notebooks)
- **Tác dụng:** Phân tích tương tác, EDA, training, evaluation
- **Vị trí:** `notebooks/`
- **Cách sử dụng:** Mở bằng Jupyter Notebook / Lab
- **Lợi ích:** Trực quan, có biểu đồ, dễ debug

### `.pkl` (Pickle - Model Files)
- **Tác dụng:** Lưu trữ mô hình ML đã huấn luyện
- **Vị trí:** `models/`
- **Cách tải:** 
  ```python
  import joblib
  model = joblib.load('models/random_forest_model.pkl')
  ```

### `.csv` (Data Files)
- **Tác dụng:** Chứa dữ liệu thô từ Kaggle
- **Vị trí:** `data/`
- **Định dạng:** 768 hàng × 9 cột

### `.md` (Markdown)
- **Tác dụng:** Tài liệu hướng dẫn
- **Vị trí:** Root directory
- **Cách xem:** Trình duyệt, VS Code, GitHub

### `.txt` (Text Files)
- **Tác dụng:** Danh sách dependencies
- **Vị trí:** Root directory (`requirements.txt`)

---

## 🔍 Cách Tìm Các Files

### Tìm theo Chức Năng

**Muốn xử lý dữ liệu:**
- `src/preprocessing.py` - Code xử lý
- `02_Model_Training.ipynb` - Ví dụ sử dụng

**Muốn huấn luyện mô hình:**
- `src/models.py` - Code huấn luyện
- `02_Model_Training.ipynb` - Notebook hoàn chỉnh

**Muốn đánh giá mô hình:**
- `src/evaluation.py` - Code đánh giá
- `03_Model_Evaluation.ipynb` - Notebook hoàn chỉnh

**Muốn dự đoán:**
- `src/demo.py` - Hàm dự đoán
- `04_Demo_Prediction.ipynb` - Demo ứng dụng

**Muốn học & hiểu dữ liệu:**
- `01_EDA.ipynb` - Phân tích chi tiết

---

## 💾 Quản Lý Dung Lượng

```
Dung lượng ước tính:
├── src/ ..................... ~50 KB
├── notebooks/ ............... ~500 KB (chưa chạy)
├── models/ .................. ~270 KB (sau khi tạo)
├── data/ .................... ~25 KB (sau khi tải)
├── results/ ................. ~1-2 MB (sau khi chạy)
├── .md files ................ ~50 KB
└── Total .................... ~2-3 MB

💡 Tip: Có thể xóa results/ để tiết kiệm dung lượng
       (sẽ được tạo lại khi chạy notebooks)
```

---

## 🆘 Troubleshooting - Tìm Files

### Không tìm thấy `diabetes.csv`
```bash
# Kiểm tra
ls data/diabetes.csv

# Nếu không tìm thấy, tải từ:
# https://www.kaggle.com/uciml/pima-indians-diabetes-database
```

### Không tìm thấy `models/*.pkl`
```bash
# Giải pháp: Chạy 02_Model_Training.ipynb
# Nó sẽ tạo các file .pkl
```

### Không tìm thấy module `src.*`
```bash
# Kiểm tra sys.path
import sys
sys.path.append('../')  # Thêm dòng này vào notebook

# Hoặc chạy từ thư mục `final/`
```

---

## 📞 Hỗ Trợ

**Cần giúp đỡ?**
- Xem `README.md` - Hướng dẫn chi tiết
- Xem `GETTING_STARTED.md` - Hướng dẫn từng bước
- Xem docstring trong `.py` files - `help(function_name)`
- Xem comments trong `.ipynb` - Chú thích chi tiết

---

## 📅 Lịch Sử

| Ngày | Sự Kiện |
|------|---------|
| 20/10/2025 | 🎉 Hoàn thành tất cả files |
| 20/10/2025 | ✅ Dự án sẵn sàng sử dụng |

---

**✨ Danh sách files hoàn chỉnh! Sẵn sàng bắt đầu dự án!**

Hãy bắt đầu với `README.md` hoặc `GETTING_STARTED.md` 🚀
