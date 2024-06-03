import os
import warnings

import joblib

base_dir = os.path.dirname(os.path.abspath(__file__))
weather_dir = os.path.dirname(base_dir)
model_dir = os.path.join(weather_dir, 'src')

warnings.filterwarnings("ignore", message="X does not have valid feature names")

def predict(outlook, temperature, humidity, windy):
    # Load model và label encoders
    clf = joblib.load(model_dir + '/decision_tree_model.joblib')
    le_outlook = joblib.load(model_dir + '/le_outlook.joblib')
    le_play = joblib.load(model_dir + '/le_play.joblib')

    # Chuyển đổi input
    input_data = [outlook, int(temperature), int(humidity), windy]
    input_data[0] = le_outlook.transform([input_data[0]])[0]
    input_data[3] = int(input_data[3])
    input_data = [input_data]

    # Dự đoán
    prediction = clf.predict(input_data)

    # Chuyển đổi kết quả về dạng ban đầu
    prediction = le_play.inverse_transform(prediction)
    
    return prediction[0]

# Ví dụ sử dụng hàm
outlook = 'sunny'
temperature = 60
humidity = 90
windy = False

prediction = predict(outlook, temperature, humidity, windy)
print(f"Dự đoán: {prediction}")
