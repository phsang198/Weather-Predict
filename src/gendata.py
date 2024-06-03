import pandas as pd
import random

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('weather.csv')

# Tạo thêm mẫu dữ liệu
new_data = []
for _ in range(100):  # Tạo 10 mẫu mới
    new_data.append({
        'outlook': random.choice(['sunny', 'overcast', 'rain']),
        'temperature': random.randint(60, 90),
        'humidity': random.randint(60, 100),
        'windy': random.choice([True, False]),
        'play': random.choice(['yes', 'no'])
    })

# Chuyển danh sách dictionaries thành DataFrame mới
new_data_df = pd.DataFrame(new_data)

# Nối DataFrame mới vào DataFrame hiện có
data = pd.concat([data, new_data_df], ignore_index=True)

# Lưu dữ liệu mới vào tệp CSV
data.to_csv('new_weather_data.csv', index=False)
