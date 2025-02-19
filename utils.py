from flask import Flask, request, jsonify
import pandas as pd
import joblib
from prophet import Prophet

def preprocess_request_data(request_data):
    df = pd.DataFrame(request_data)

    if not all(col in df.columns for col in ['date', 'time_block', 'bed']):
        raise ValueError("Missing required columns in the request data")

    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    
    df['ds'] = df['date'] + pd.to_timedelta(df['time_block'], unit='h')
    
    df = df[['ds', 'time_block', 'bed']]
    df.columns = ['ds', 'time_block', 'bed']
    
    return df

def preprocess_and_split_data(df, period=42, test_days=28):
    df = df.sort_values(by='ds')
    start_date = df['ds'].min()
    end_date = df['ds'].max()
    
    cycles = []

    while start_date + pd.Timedelta(days=period) <= end_date:
        cycle_end_date = start_date + pd.Timedelta(days=period)
        cycle_df = df[(df['ds'] >= start_date) & (df['ds'] < cycle_end_date)]
        test_df = cycle_df[cycle_df['ds'] > cycle_end_date - pd.Timedelta(days=test_days)]
        
        cycles.append(test_df)
        
        start_date = cycle_end_date

    return cycles

def train_prophet(df):
    model = Prophet()
    model.add_regressor('bed')
    model.add_regressor('time_block')
    
    model.fit(df)
    
    return model

def make_forecast(model, request_data, periods=42, freq='4H'):
    future = model.make_future_dataframe(request_data, periods=periods, freq=freq)
    
    future['bed'] = request_data['bed'].iloc[-1]  
    future['time_block'] = future['ds'].dt.hour 
    
    forecast = model.predict(future)
    forecast_result = forecast[['ds', 'yhat']]
    return forecast_result.to_dict(orient='records')

def save_trained_model(model, save_path="prophet_model.pkl"):
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

def load_trained_model(load_path="prophet_model.pkl"):
    model = joblib.load(load_path)
    print(f"Model loaded from {load_path}")
    return model
