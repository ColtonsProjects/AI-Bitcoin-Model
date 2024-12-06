import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.model_selection import train_test_split
import xgboost as xgb
from prophet import Prophet
from datetime import datetime


# Example function to preprocess data
def preprocess_data(prices):
    df = pd.DataFrame(prices)
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})  # Ensure all columns except 'open_time' are float
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close', 'high', 'low', 'open', 'volume']])
    df[['close', 'high', 'low', 'open', 'volume']] = scaled_data
    return df, scaler

# Example function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    data_array = np.array(data)
    
    # Ensure data is 2D if it isn't already
    if len(data_array.shape) == 1:
        data_array = data_array.reshape(-1, 1)
        
    for i in range(len(data_array) - seq_length):
        sequences.append(data_array[i:i+seq_length])
    return np.array(sequences)

# Example function to build and train an LSTM model
def build_and_train_model(data):
    seq_length = 30
    n_features = 5  # close, high, low, open, volume
    
    # Prepare the data
    features = data[['close', 'high', 'low', 'open', 'volume']].values
    
    # Create sequences manually with explicit shape checking
    X = []
    y = []
    for i in range(len(features) - seq_length):
        sequence = features[i:(i + seq_length)]
        if sequence.shape != (seq_length, n_features):
            sequence = sequence.reshape(seq_length, n_features)
        target = features[i + seq_length, 0]  # predict close price
        X.append(sequence)
        y.append(target)
    
    # Convert to numpy arrays and verify shapes
    X = np.array(X)
    y = np.array(y)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")  # Debug print
    
    if len(X.shape) != 3 or X.shape[1:] != (seq_length, n_features):
        X = X.reshape(-1, seq_length, n_features)
    
    # Split data
    if len(X) < 2:
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training shapes - X: {X_train.shape}, y: {y_train.shape}")  # Debug print
    
    # Build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, n_features)))
    model.add(LSTM(50))
    model.add(Dense(1))
    
    # Compile
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Fit with error checking
    try:
        model.fit(
            X_train, 
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
    except Exception as e:
        print(f"Error during training: {e}")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        raise e
    
    return model

# To make predictions
def predict_next_price(model, data, scaler):
    seq_length = 30
    n_features = 5
    
    # Get the last sequence
    features = data[['close', 'high', 'low', 'open', 'volume']].values
    
    # Ensure we have enough data
    if len(features) < seq_length:
        raise ValueError(f"Not enough data points. Need at least {seq_length}")
    
    # Get last sequence and ensure correct shape
    last_sequence = features[-seq_length:]
    if last_sequence.shape != (seq_length, n_features):
        last_sequence = last_sequence.reshape(seq_length, n_features)
    
    # Add batch dimension
    X = np.expand_dims(last_sequence, axis=0)  # Shape: (1, seq_length, n_features)
    print(f"Prediction input shape: {X.shape}")  # Debug print
    
    # Make prediction
    prediction = model.predict(X, verbose=0)
    
    # Prepare for inverse transform
    prediction = prediction.reshape(-1, 1)
    zeros = np.zeros((prediction.shape[0], 4))
    
    # Inverse transform
    try:
        inverse_prediction = scaler.inverse_transform(np.hstack((prediction, zeros)))
        return inverse_prediction[:, 0]
    except Exception as e:
        print(f"Error during inverse transformation: {e}")
        return None

def build_and_train_xgboost(data):
    seq_length = 30  # Fixed sequence length
    feature_count = 5  # close, high, low, open, volume
    
    # Create sequences with fixed dimensions
    sequences = create_sequences(data[['close', 'high', 'low', 'open', 'volume']].values, seq_length)
    X = sequences[:, :-1, :].reshape(sequences.shape[0], (seq_length - 1) * feature_count)  # Should be 145 features
    y = sequences[:, -1, 0]  # Predict the closing price
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=3
    )
    model.fit(X, y)
    
    return model

def build_and_train_prophet(data):
    # Prophet requires specific column names: 'ds' for dates and 'y' for values
    prophet_data = pd.DataFrame({
        'ds': pd.to_datetime(data.index),
        'y': data['close'].values
    })
    
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_data)
    
    return model

def predict_next_price_xgboost(model, data, scaler):
    seq_length = 30  # Same as in training
    feature_count = 5  # Same number of features as training
    
    # Get the last sequence
    last_data = data[['close', 'high', 'low', 'open', 'volume']].values[-(seq_length-1):]
    last_sequence = last_data.reshape(1, (seq_length-1) * feature_count)  # Reshape to match training dimensions
    
    prediction = model.predict(last_sequence)
    prediction = prediction.reshape(-1, 1)
    
    # Inverse transform the prediction
    zeros = np.zeros((prediction.shape[0], 4))
    try:
        inverse_prediction = scaler.inverse_transform(np.hstack((prediction, zeros)))
        return inverse_prediction[:, 0]
    except Exception as e:
        print(f"Error during inverse transformation: {e}")
        return None

def predict_next_price_prophet(model, data):
    # Create future dates dataframe
    future = model.make_future_dataframe(periods=1)
    
    # Make prediction
    forecast = model.predict(future)
    
    # Get the last predicted value
    prediction = forecast['yhat'].iloc[-1]
    return prediction
