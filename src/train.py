import os
import pickle

import pandas as pd
from C45 import C45Classifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Xác định đường dẫn thư mục
base_dir = os.path.dirname(os.path.abspath(__file__))
weather_dir = os.path.dirname(base_dir)
data_dir = os.path.join(weather_dir, 'data')
model_dir = os.path.join(weather_dir, 'model')

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv(data_dir +'/weather.csv')

# Kiểm tra dữ liệu
print(data.head())

X = data.drop(['play'], axis=1)  # Loại bỏ cột 'play' (target) khỏi features
y = data['play'] #target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình cây quyết định và huấn luyện
model = C45Classifier()  
model.fit(X_train, y_train)

# Lưu mô hình
with open(model_dir + '/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("Độ chính xác:", metrics.accuracy_score(y_test, y_pred))
