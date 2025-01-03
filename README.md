<<<<<<< HEAD
# Hướng Dẫn Chạy Dự Án
## 1. Yêu Cầu Môi Trường
- **Python**: >= 3.8
- Git-2.47.1-64-bit.exe
- git-lfs-windows-v3.6.0.exe
- Các thư viện cần thiết:

pip install pandas numpy scikit-learn matplotlib seaborn

## 2. Cấu Trúc Thư Mục
random_forest_project/
├── main.py                    # Dự đoán và xuất kết quả
├── notebook/
│   └── train_model.ipynb      # Huấn luyện mô hình
├── data/
│   ├── train.csv              # Dữ liệu huấn luyện
│   ├── test.csv               # Dữ liệu kiểm tra
├── model/
│   └── random_forest_model.pkl # Mô hình đã huấn luyện
└── result/
    └── test_predictions.csv   # Kết quả dự đoán

## 3. Hướng Dẫn Chạy
###########################

git clone https://github.com/PhuongNGT/NHOM6-NT211.git



###########################
### 3.1. Chạy Notebook Huấn Luyện
1. Mở terminal và di chuyển vào thư mục gốc của dự án:

cd NHOM6-NT211

2. Khởi chạy Jupyter Notebook:

jupyter notebook

3. Mở file `notebook/train_model.ipynb` và chạy lần lượt các ô (cell) để:
- Tiền xử lý dữ liệu.
- Huấn luyện mô hình.
- Lưu mô hình tại `model/random_forest_model.pkl`.
- luu file 'label_encoder.pkl'  tại `label_encoder.pkl'`.

### 3.2. Chạy File `main.py` Để Dự Đoán
1. Đảm bảo file mô hình `random_forest_model.pkl` đã tồn tại.
2. Đảm bảo file 'label_encoder.pkl' đã tồn tại.
3. Di chuyển đến thư mục gốc của dự án
4. Chạy file `main.py`:

### 3.3. Kết Quả
- **Kết quả dự đoán** được lưu tại: result/test_predictions.csv

- **Nội dung file kết quả**:
Bao gồm các cột từ tập `test.csv` và cột mới `Predicted_Label` chứa nhãn dự đoán.


