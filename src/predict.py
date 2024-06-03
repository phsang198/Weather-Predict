import os
import pickle
import warnings

import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
weather_dir = os.path.dirname(base_dir)
model_dir = os.path.join(weather_dir, 'model')

warnings.filterwarnings("ignore", message="X does not have valid feature names")

def predict(outlook, temperature, humidity, windy):
    # Load model 

    model = pickle.load(open(model_dir + '/model.pkl', 'rb'))

    #  input
    input_data = pd.DataFrame({
    'outlook': [outlook],
    'temperature': [int(temperature)],
    'humidity': [int(humidity)],
    'windy': [windy]
    })

    # Dự đoán
    prediction = model.predict(input_data)

    return prediction[0]

# Ví dụ sử dụng hàm
outlook = 'sunny'
temperature = 85
humidity = 90
windy = True

prediction = predict(outlook, temperature, humidity, windy)
print(f"Dự đoán: {prediction}")
