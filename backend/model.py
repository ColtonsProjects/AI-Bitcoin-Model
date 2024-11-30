import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.model_selection import train_test_split


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
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

# Example function to build and train an LSTM model
def build_and_train_model(data):
    seq_length = min(30, len(data) - 1)  # Ensure at least one sequence can be created

    if len(data) < seq_length:
        raise ValueError(f"Not enough data points to create sequences of length {seq_length}")

    sequences = create_sequences(data[['close', 'high', 'low', 'open', 'volume']].values, seq_length)
    X, y = sequences[:, :-1, :], sequences[:, -1, 0]  # Use all features for X, only 'close' for y

    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    if len(X) < 2:
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    return model

# To make predictions
def predict_next_price(model, data, scaler):
    if len(data) < 30:
        raise ValueError("Not enough data to create a sequence of length 30")

    # Exclude 'open_time' from the data used for prediction
    last_sequence = data[['close', 'high', 'low', 'open', 'volume']].values[-30:].reshape(1, 30, 5)  # 5 features

    # Ensure the prediction is a NumPy array
    prediction = model.predict(last_sequence)
    if not isinstance(prediction, np.ndarray):
        prediction = np.array(prediction)

    # Reshape the prediction to ensure it's 2D for inverse transform
    prediction = prediction.reshape(-1, 1)

    # Create a zero array with the same number of columns as the original data minus the target column
    zeros = np.zeros((prediction.shape[0], 4))  # 4 because we have 5 features and 1 is the target

    # Concatenate the prediction with zeros to match the original feature space
    try:
        inverse_prediction = scaler.inverse_transform(np.hstack((prediction, zeros)))
        return inverse_prediction[:, 0]  # Return only the 'close' price prediction
    except Exception as e:
        print(f"Error during inverse transformation: {e}")
        return None
