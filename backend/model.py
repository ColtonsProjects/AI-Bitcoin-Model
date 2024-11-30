import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


# Example function to preprocess data
def preprocess_data(prices):
    df = pd.DataFrame(prices)
    df['close'] = df['close'].astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['close'] = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    return df, scaler

# Example function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

# Example function to build and train an LSTM model
def build_and_train_model(data):
    # Adjust the sequence length based on available data
    seq_length = min(30, len(data['close'].values) - 1)  # Ensure at least one sequence can be created

    # Check if there are enough data points
    if len(data['close'].values) < seq_length:
        raise ValueError(f"Not enough data points to create sequences of length {seq_length}")

    # Create sequences
    sequences = create_sequences(data['close'].values, seq_length)
    print("Sequences Shape:", sequences.shape)  # Debug: Check sequences shape

    # Split sequences into features and target
    X, y = sequences[:, :-1], sequences[:, -1]
    print("X Shape:", X.shape)  # Debug: Check X shape
    print("y Shape:", y.shape)  # Debug: Check y shape

    # Reshape X for LSTM input
    X = X.reshape((X.shape[0], X.shape[1], 1))
    print("Reshaped X Shape:", X.shape)  # Debug: Check reshaped X shape

    # Use all data for training if not enough for split
    if len(X) < 2:
        print("Not enough sequences to perform train/test split, using all data for training")
        X_train, y_train = X, y
        X_test, y_test = X, y  # This is just to avoid errors, not for actual testing
    else:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("X_train Shape:", X_train.shape)  # Debug: Check X_train shape
        print("X_test Shape:", X_test.shape)  # Debug: Check X_test shape
        print("y_train Shape:", y_train.shape)  # Debug: Check y_train shape
        print("y_test Shape:", y_test.shape)  # Debug: Check y_test shape

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    return model

# To make predictions
def predict_next_price(model, data, scaler):
    # Ensure the last sequence is reshaped correctly
    if len(data['close'].values) < 30:
        raise ValueError("Not enough data to create a sequence of length 30")

    last_sequence = data['close'].values[-30:].reshape(1, -1, 1)  # Reshape to 3D array for LSTM input
    print("Last Sequence Shape:", last_sequence.shape)  # Debug: Check last sequence shape

    # Make prediction
    prediction = model.predict(last_sequence)
    print("Prediction Shape:", prediction.shape)  # Debug: Check prediction shape

    # Reshape prediction for inverse transform
    prediction = prediction.reshape(-1, 1)  # Ensure it's 2D for inverse transform

    # Inverse transform the prediction
    inverse_prediction = scaler.inverse_transform(prediction)
    print("Inverse Prediction:", inverse_prediction)  # Debug: Check inverse prediction

    return inverse_prediction
