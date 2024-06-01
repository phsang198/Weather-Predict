
# Chuyển đổi đầu vào thành dạng số
input_data = ['rain', 60, 60, True]
input_data[0] = le_outlook.transform([input_data[0]])[0]  # Chuyển đổi 'rain' thành số
input_data[3] = int(input_data[3])  # Chuyển đổi True thành 1, False thành 0

# Dự đoán với đầu vào đã chuẩn bị
predicted = clf.predict([input_data])

# Giải mã kết quả từ số về chuỗi ('yes' hoặc 'no')
predicted_label = le_play.inverse_transform(predicted)[0]

print("Dự đoán thời tiết:", predicted_label)