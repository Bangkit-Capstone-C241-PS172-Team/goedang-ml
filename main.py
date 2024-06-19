import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

WINDOW_SIZE = 30
STEP_AHEAD = 7
THRESHOLD = 20

def load_model(name):
    path=f"models/model_{name}.h5"
    model = tf.keras.models.load_model(path)
    scaler = pickle.load(open(f"./models/scaler_{name}.pkl", 'rb'))
    return model, scaler

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        data = request.get_json()
        
        if 'item' not in data or 'harga' not in data or 'tanggal' not in data:
            return jsonify({"message": "Data tidak lengkap"}), 400
        
        item = data["item"]
        harga = data["harga"]
        dates = data["tanggal"]
        
        item = item.replace(" ", "_").lower()
        
        model, scaler = load_model(item)

        date_series = pd.to_datetime(dates, format='mixed')
        date_difference  = (date_series.max() - date_series.min()).days
        
        if (date_difference < WINDOW_SIZE - 1) or (len(harga) < THRESHOLD or len(dates) < THRESHOLD):
            return jsonify({"message": "Data tidak cukup untuk melakukan forecasting"})
        
        df = pd.DataFrame({
            'tanggal': date_series,
            'harga': harga
        })
        
        df['harga'] = df['harga'].astype(float)

        df = df.sort_values(by='tanggal')
        df.set_index('tanggal', inplace=True)
        
        # Menentukan rentang tanggal lengkap
        all_dates = pd.date_range(start=df.index.min(), end=df.index.max())
        
        df = df.groupby('tanggal').agg({
            'harga': 'mean',
        })
        
        # Mengulang data agar sesuai dengan rentang tanggal lengkap
        df = df.reindex(all_dates)
        df['harga'] = df['harga'].interpolate()
        
        # Mengambil data harga 30 hari terakhir
        data_harga = df[['harga']].values
        last_window = np.array(data_harga[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
        
        print(last_window)
        
        forecast = []
        input_data = last_window
        
        for _ in range(STEP_AHEAD):
            prediction = model.predict(input_data)
            forecast.append(prediction[0, 0])
            input_data = np.append(input_data[:,1:,:], prediction.reshape(1,1,1), axis=1)
        
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten().tolist()
        first_date = pd.to_datetime(dates[-1]) + pd.DateOffset(days=1)
        next_dates = pd.date_range(start=first_date, periods=STEP_AHEAD)
        
        response = {
            "item": item,
            "forecast": forecast,
            "dates": next_dates.strftime("%Y-%m-%d").tolist()
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"message": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))