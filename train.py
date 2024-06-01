import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('weather.csv')

# Kiểm tra dữ liệu
print(data.head())

# Chuyển đổi các cột kiểu chuỗi thành số
le_outlook = LabelEncoder()
le_play = LabelEncoder()

data['outlook'] = le_outlook.fit_transform(data['outlook'])
data['play'] = le_play.fit_transform(data['play'])
data['windy'] = data['windy'].astype(int)

# Xác định biến đầu vào (features) và biến mục tiêu (target)
X = data[['outlook', 'temperature', 'humidity', 'windy']]
y = data['play']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Tạo mô hình cây quyết định
clf = DecisionTreeClassifier(criterion='entropy')  # C4.5 sử dụng entropy
clf = clf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test)

# Đánh giá mô hình
print("Độ chính xác:", metrics.accuracy_score(y_test, y_pred))

# Hiển thị cây quyết định
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=le_play.classes_)
plt.show()
