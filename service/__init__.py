import pandas as pd
from prophet import Prophet
import joblib
from flask import Flask, request, jsonify
import os

MODEL_PATH = "prophet_model.pkl"

def preprocess_data(file_path="Q1-Data.csv"):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values(by=['date', 'time_block']).reset_index(drop=True)
    start_date = df['date'].min()
    cycles = []
    while start_date + pd.Timedelta(days=42) <= df['date'].max():
        cycle_data = df[(df['date'] >= start_date) & (df['date'] < start_date + pd.Timedelta(days=42))]
        last_28_days = cycle_data.tail(28 * 6)
        cycles.append(last_28_days)
        start_date += pd.Timedelta(days=42)
    return pd.concat(cycles).reset_index(drop=True)

def train_prophet(df):
    df = df.rename(columns={"date": "ds", "total_patient_volume": "y"})
    df["ds"] = pd.to_datetime(df["ds"])

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        model.fit(df[['ds', 'y']])
    else:
        model = Prophet()
        model.fit(df[['ds', 'y']])

    joblib.dump(model, MODEL_PATH)


def load_trained_model():
    return joblib.load("prophet_model.pkl")

def preprocess_request_data(request_data):
    df = pd.DataFrame(request_data)
    if not all(col in df.columns for col in ['date', 'time_block', 'bed']):
        raise ValueError("Thiếu cột dữ liệu cần thiết")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    df['ds'] = df['date'] + pd.to_timedelta(df['time_block'], unit='h')
    return df[['ds', 'time_block', 'bed']]

def make_forecast(model, request_data, periods=42 * 6, freq='4H'):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    future['bed'] = request_data['bed'].iloc[-1]
    future['time_block'] = future['ds'].dt.hour
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].to_dict(orient='records')

def create_app():
    app = Flask(__name__)
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            request_data = request.get_json()
            data = preprocess_request_data(request_data)
            model = load_trained_model()
            forecast = make_forecast(model, data)
            return jsonify({"forecast": forecast}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        
    @app.route("/train", methods=["POST"])
    def train():
        try:
            request_data = request.get_json()
            if not request_data:
                return jsonify({"error": "No data provided for training"}), 400

            df = pd.DataFrame(request_data)
            if "date" not in df.columns or "total_patient_volume" not in df.columns:
                return jsonify({"error": "Missing required columns: 'date', 'total_patient_volume'"}), 400

            train_prophet(df)
            return jsonify({"message": "Model has been successfully updated!"}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return app

