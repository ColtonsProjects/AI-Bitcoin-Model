import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

BASE_URL = "http://backend:5001"

def fetch_data():
    endpoints = [
        "/open-interest",
        "/funding-rate",
        "/liquidations",
        "/price",
        "/hour-price",
    ]
    
    data = {}
    for endpoint in endpoints:
        response = requests.get(BASE_URL + endpoint)
        if response.status_code == 200:
            data[endpoint] = response.json()
        else:
            print(f"Failed to fetch data from {endpoint}")
    
    return data

def create_dataframe(data):
    df_list = []
    
    if "/month-price" in data:
        month_prices = data["/month-price"]
        df_month = pd.DataFrame(month_prices)
        df_list.append(df_month)
    
    if "/week-price" in data:
        week_prices = data["/week-price"]
        df_week = pd.DataFrame(week_prices)
        df_list.append(df_week)
    
    if df_list:
        combined_df = pd.concat(df_list, axis=0, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Assuming the first column is the target
    return np.array(x), np.array(y)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train, epochs=50, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

def evaluate_model(model, x_test, y_test, scaler):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    print(f'RMSE: {rmse}')
    return predictions, rmse

def prepare_and_train_model():
    data = create_dataframe(fetch_data())
    scaled_data, scaler = preprocess_data(data)
    
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    seq_length = 60
    x_train, y_train = create_sequences(train_data, seq_length)
    x_test, y_test = create_sequences(test_data, seq_length)
    
    model = build_model((x_train.shape[1], x_train.shape[2]))
    train_model(model, x_train, y_train)
    
    predictions, rmse = evaluate_model(model, x_test, y_test, scaler)
    return model, scaler, seq_length

# Call this function to prepare and train the model
model, scaler, seq_length = prepare_and_train_model()