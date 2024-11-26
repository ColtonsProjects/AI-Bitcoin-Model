# import numpy as np
# import pandas as pd
# import json
# import os
# import requests
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout

# BASE_URL = "http://backend:5001"
# DATA_FILE = "fetched_data.json"

# def wait_for_server():
#     health_url = f"{BASE_URL}/health"
#     print("Waiting for the server to be ready...", flush=True)
#     while True:
#         try:
#             response = requests.get(health_url)
#             if response.status_code == 200 and response.json().get("status") == "ok":
#                 print("Server is up and running.", flush=True)
#                 break
#         except requests.exceptions.ConnectionError:
#             pass  # Server is not up yet
#         except Exception as e:
#             print(f"Unexpected error: {e}", flush=True  )
#         time.sleep(5)  # Wait 5 seconds before retrying

# # Fetch data and save locally
# def fetch_data():
#     endpoints = [
#         "/open-interest", "/funding-rate", "/liquidations",
#         "/price", "/month-price", "/four-hour-price",
#         "/day-price", "/hour-price", "/week-price"
#     ]

#     data = {}
#     for endpoint in endpoints:
#         response = requests.get(BASE_URL + endpoint)
#         if response.status_code == 200:
#             data[endpoint] = response.json()
#         else:
#             print(f"Failed to fetch data from {endpoint}", flush=True)

#     # Save fetched data locally
#     with open(DATA_FILE, "w") as file:
#         json.dump(data, file)

#     return data

# # Load saved data if exists
# def load_data():
#     if os.path.exists(DATA_FILE):
#         with open(DATA_FILE, "r") as file:
#             return json.load(file)
#     return fetch_data()

# # Parse data from endpoints
# def parse_data(raw_data):
#     parsed_data = {}

#     # Parse each endpoint
#     if "/open-interest" in raw_data:
#         parsed_data["open_interest"] = pd.DataFrame(raw_data["/open-interest"])

#     if "/funding-rate" in raw_data:
#         parsed_data["funding_rate"] = pd.DataFrame(raw_data["/funding-rate"])

#     if "/liquidations" in raw_data:
#         history = raw_data["/liquidations"].get("history", [])
#         parsed_data["liquidations"] = pd.DataFrame(history)

#     for key in ["/price", "/month-price", "/four-hour-price", "/day-price", "/hour-price", "/week-price"]:
#         if key in raw_data:
#             parsed_data[key.strip("/")] = pd.DataFrame(raw_data[key])

#     return parsed_data

# # Combine data into a unified DataFrame
# def create_dataframe(parsed_data):
#     # Example: Combine month-price and week-price
#     df_list = [parsed_data.get("month-price"), parsed_data.get("week-price")]
#     combined_df = pd.concat(df_list, axis=0, ignore_index=True) if df_list else pd.DataFrame()
#     return combined_df

# # Preprocess data for training
# def preprocess_data(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data)
#     return scaled_data, scaler

# # Create sequences for LSTM
# def create_sequences(data, seq_length):
#     x, y = [], []
#     for i in range(len(data) - seq_length):
#         x.append(data[i:i + seq_length])
#         y.append(data[i + seq_length, 0])  # Assuming the first column is the target
#     return np.array(x), np.array(y)

# # Build the LSTM model
# def build_model(input_shape):
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=input_shape),
#         Dropout(0.2),
#         LSTM(50, return_sequences=False),
#         Dropout(0.2),
#         Dense(1)
#     ])
#     model.compile(optimizer="adam", loss="mean_squared_error")
#     return model

# # Train the model
# def train_model(model, x_train, y_train, epochs=50, batch_size=32):
#     model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# # Evaluate and save results
# def evaluate_model(model, x_test, y_test, scaler):
#     predictions = model.predict(x_test)
#     predictions = scaler.inverse_transform(predictions)
#     rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
#     print(f"RMSE: {rmse}")
#     return predictions, rmse

# # Main pipeline
# def prepare_and_train_model():
#     raw_data = load_data()
#     parsed_data = parse_data(raw_data)
#     combined_df = create_dataframe(parsed_data)

#     if combined_df.empty:
#         print("No valid data to process.")
#         return

#     scaled_data, scaler = preprocess_data(combined_df.values)
#     seq_length = 60

#     train_size = int(len(scaled_data) * 0.8)
#     train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

#     x_train, y_train = create_sequences(train_data, seq_length)
#     x_test, y_test = create_sequences(test_data, seq_length)

#     model = build_model((x_train.shape[1], x_train.shape[2]))
#     train_model(model, x_train, y_train)

#     predictions, rmse = evaluate_model(model, x_test, y_test, scaler)

#     # Save model and scaler
#     model.save("bitcoin_price_model.h5")
#     with open("scaler.pkl", "wb") as f:
#         pickle.dump(scaler, f)

#     return model, scaler, seq_length

# # Start the pipeline
# if __name__ == "__main__":
#     wait_for_server()
#     model, scaler, seq_length = prepare_and_train_model()
