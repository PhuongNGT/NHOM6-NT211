# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def align_features(model, X):
    if hasattr(model, 'n_features_in_'):
        required_columns = model.n_features_in_
        if X.shape[1] != required_columns:
            print(f"Số lượng đặc trưng không khớp! Dữ liệu có {X.shape[1]} cột, mô hình yêu cầu {required_columns} cột.")
            
            # Thêm các cột còn thiếu với giá trị mặc định 0
            missing_columns = required_columns - X.shape[1]
            if missing_columns > 0:
                print(f"Đang thêm {missing_columns} cột còn thiếu với giá trị 0...")
                X = np.pad(X, ((0, 0), (0, missing_columns)), mode='constant', constant_values=0)
    return X

# Đọc tập test
test_data_path = 'dataset/test.csv'
test_data = pd.read_csv(test_data_path)

# Tiền xử lý tập test
X_test = pd.get_dummies(test_data.copy())
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Tải mô hình
model_save_path = 'model/random_forest_model.pkl'
try:
    with open(model_save_path, 'rb') as f:
        model = pickle.load(f)
    print("Mô hình đã được tải lại từ đĩa.")
except FileNotFoundError:
    print("Không tìm thấy mô hình đã lưu.")
    exit()

# Đồng bộ đặc trưng
X_test = align_features(model, X_test)
label_encoder = LabelEncoder()
# Dự đoán trên tập test
if 'Label' in test_data.columns:
    
    test_data['Original_Label'] = test_data['Label']
    test_data['Label'] = label_encoder.fit_transform(test_data['Label'])
    y_pred_test = model.predict(X_test)
    test_data['Predicted_Label'] = label_encoder.inverse_transform(y_pred_test)
    test_data['Label'] = label_encoder.inverse_transform(test_data['Label'])

    # Đánh giá
    test_accuracy = accuracy_score(test_data['Original_Label'], test_data['Predicted_Label'])
    test_precision = precision_score(test_data['Original_Label'], test_data['Predicted_Label'], average='weighted')
    test_recall = recall_score(test_data['Original_Label'], test_data['Predicted_Label'], average='weighted')
    test_f1 = f1_score(test_data['Original_Label'], test_data['Predicted_Label'], average='weighted')

    print(f"\nĐộ chính xác trên tập test: {test_accuracy:.4f}")
    print(f"Độ chính xác (Precision): {test_precision:.4f}")
    print(f"Độ nhạy (Recall): {test_recall:.4f}")
    print(f"F1-score: {test_f1:.4f}")

    print("\nBáo cáo phân loại trên tập test:")
    print(classification_report(test_data['Original_Label'], test_data['Predicted_Label'], target_names=label_encoder.classes_))

    # Ma trận nhầm lẫn
    conf_matrix_test = confusion_matrix(test_data['Original_Label'], test_data['Predicted_Label'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Ma trận nhầm lẫn trên tập test')
    plt.xlabel('Nhãn dự đoán')
    plt.ylabel('Nhãn thực tế')
    plt.show()
else:
    y_pred_test = model.predict(X_test)

    # Nếu `LabelEncoder` đã được lưu trước đó, tải lại
    try:
        with open('model/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError:
        print("Không tìm thấy tệp label_encoder.pkl. Đảm bảo bạn đã lưu LabelEncoder khi huấn luyện.")
        exit()

    # Chuyển đổi nhãn số thành nhãn ban đầu
    test_data['Predicted_Label'] = label_encoder.inverse_transform(y_pred_test)
    print("Tập test không có cột Label, chỉ hiển thị nhãn dự đoán.")

# Lưu kết quả dự đoán
output_file = 'result/test_predictions_rf.csv'
test_data.to_csv(output_file, index=False)
print(f"Kết quả dự đoán đã được lưu tại {output_file}")

