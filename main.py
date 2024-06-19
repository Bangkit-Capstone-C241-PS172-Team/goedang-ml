import tensorflow as tf
import numpy as np
import os
import pandas as pd
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)


WINDOW_SIZE = 10
STEP_AHEAD = 7

def load_model(name):
    path= os.path.join("models", f"model_{name}.h5")
    model = tf.keras.models.load_model(path)
    scaler = pickle.load(open(os.path.join("models", f"scaler_{name}.pkl"), 'rb'))
    return model, scaler

model_cmbtt, scaler_cmbtt = load_model("cmbtt") 
model_cmktt, scaler_cmktt = load_model("cmktt")
model_crmtt, scaler_crmtt = load_model("crmtt")
model_crptt, scaler_crptt = load_model("crptt")

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        data = request.get_json()
        harga = data["harga"]
        dates = data["tanggal"]
        
        if len(harga) < WINDOW_SIZE:
            return jsonify({"message": "Data tidak cukup untuk melakukan forecasting"})
        
        last_window = np.array(harga[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE,1)
        forecast = []
        input_data = last_window
        
        for _ in range(STEP_AHEAD):
            prediction = model_cmbtt.predict(input_data)
            forecast.append(prediction[0,0])
            input_data = np.append(input_data[:,1:,:], prediction.reshape(1,1,1), axis=1)
        
        forecast =scaler_cmbtt.inverse_transform(np.array(forecast).reshape(-1,1)).flatten().tolist()
        next_dates = pd.date_range(start=dates[-1],periods=STEP_AHEAD)
        
        response = {
            "forecast": forecast,
            "dates": next_dates.strftime("%Y-%m-%d").tolist()
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"message": str(e)}) 
            
        

if __name__ == '__main__':
    app.run()