import os
import requests
import time
import datetime
from model import preprocess_data, build_and_train_model, predict_next_price, build_and_train_poly_model, predict_next_poly_price, build_and_train_lasso_model, predict_next_lasso_price
from flask import Flask, jsonify
import numpy as np
#from dotenv import load_dotenv
from flask_cors import CORS
import pandas as pd
import json
from pathlib import Path
from prediction_logger import PredictionLogger
import threading
from model_accuracy_tracker import ModelAccuracyTracker


# load_dotenv()

app = Flask(__name__)
CORS(app)

# ______________________________ Coinalyze API ______________________________

# def fetch_open_interest():
#     api_key = os.getenv('COINALYZE_API_KEY')
#     url = "https://api.coinalyze.net/v1/open-interest"
#     params = {
#         'symbols': 'BTCUSD_PERP.A', 
#         'convert_to_usd': 'true'
#     }
#     headers = {
#         'api_key': f'{api_key}'
#     }
#     response = requests.get(url, headers=headers, params=params)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return {"error": "Failed to fetch data", "status_code": response.status_code}

# def fetch_funding_rate():
#     api_key = os.getenv('COINALYZE_API_KEY')
#     url = "https://api.coinalyze.net/v1/funding-rate"
#     params = {
#         'symbols': 'BTCUSD_PERP.A' 
#     }
#     headers = {
#         'api_key': f'{api_key}'
#     }
#     response = requests.get(url, headers=headers, params=params)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return {"error": "Failed to fetch data", "status_code": response.status_code}
    
# def fetch_liquidation_data():
    # api_key = os.getenv('COINALYZE_API_KEY')
    # url = "https://api.coinalyze.net/v1/liquidation-history"
    # symbol = 'BTCUSD_PERP.A'  # Ensure this matches Coinalyze's symbol for the desired contract

    # # Define the time interval and range
    # interval = '1hour'  # Options: '1min', '5min', '15min', '30min', '1hour', '2hour', '4hour', '6hour', '12hour', 'daily'
    # current_time = int(time.time())
    # one_week_ago = current_time - 7 * 24 * 60 * 60  # 7 days ago

    # params = {
    #     'symbols': symbol,
    #     'interval': interval,
    #     'from': one_week_ago,
    #     'to': current_time,
    #     'convert_to_usd': 'true'  # Convert values to USD
    # }

    # headers = {
    #     'api_key': api_key
    # }

    # try:
    #     response = requests.get(url, headers=headers, params=params)
    #     response.raise_for_status()
    #     return response.json()
    # except requests.exceptions.HTTPError as http_err:
    #     return {"error": f"HTTP error occurred: {http_err}", "status_code": response.status_code}
    # except Exception as err:
    #     return {"error": f"An error occurred: {err}"}
# __________________________________________________________________________

# ______________________________ Binance API _______________________________

def get_price_from_binance():
    url = 'https://api.binance.us/api/v3/ticker/price'
    params = {
        'symbol': 'BTCUSDT'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return float(data['price'])
    else:
        raise Exception('API request failed with status code {}'.format(response.status_code))

def get_month_prices(interval='1d', days=30):
    url = 'https://api.binance.us/api/v3/klines'
    symbol = 'BTCUSDT'
    end_time = int(datetime.datetime.now().timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)  

    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = []
        for kline in data:
            prices.append({
                'open_time': datetime.datetime.fromtimestamp(kline[0] / 1000),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        return prices
    else:
        raise Exception(f'API request failed with status code {response.status_code}')

def get_week_prices(interval='12h', days=7):
    url = 'https://api.binance.us/api/v3/klines'
    symbol = 'BTCUSDT'
    end_time = int(datetime.datetime.now().timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)  

    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = []
        for kline in data:
            prices.append({
                'open_time': datetime.datetime.fromtimestamp(kline[0] / 1000),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        return prices
    else:
        raise Exception(f'API request failed with status code {response.status_code}')

def get_four_hour_prices(interval='15m'):
    url = 'https://api.binance.us/api/v3/klines'
    symbol = 'BTCUSDT'
    end_time = int(datetime.datetime.now().timestamp() * 1000)
    start_time = end_time - (60 * 60 * 4000)  # Past hour in milliseconds

    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 48  # 12 intervals of 5 minutes each in an hour
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = []
        for kline in data:
            prices.append({
                'open_time': datetime.datetime.fromtimestamp(kline[0] / 1000),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        return prices
    else:
        raise Exception(f'API request failed with status code {response.status_code}')

def get_hour_prices(interval='5m'):
    url = 'https://api.binance.us/api/v3/klines'
    symbol = 'BTCUSDT'
    end_time = int(datetime.datetime.now().timestamp() * 1000)
    start_time = end_time - (60 * 60 * 4000)  # Past hour in milliseconds

    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 48  # 12 intervals of 5 minutes each in an hour
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = []
        for kline in data:
            prices.append({
                'open_time': datetime.datetime.fromtimestamp(kline[0] / 1000),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        return prices
    else:
        raise Exception(f'API request failed with status code {response.status_code}')

def get_day_prices():
    url = 'https://api.binance.us/api/v3/klines'
    symbol = 'BTCUSDT'
    interval = '1h'  # 1-hour intervals
    end_time = int(datetime.datetime.now().timestamp() * 1000)
    start_time = end_time - (24 * 60 * 60 * 1000)  # Past 24 hours in milliseconds

    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 24  # 24 intervals of 1 hour each in a day
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = []
        for kline in data:
            prices.append({
                'open_time': datetime.datetime.fromtimestamp(kline[0] / 1000),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        return prices
    else:
        raise Exception(f'API request failed with status code {response.status_code}')



# __________________________________________________________________________

@app.route("/")
def home():
    return "Backend is up and running!"

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

# ______________________________ Test Coinalyze API ______________________________
@app.route("/open-interest")
def open_interest():
    data = fetch_open_interest()
    return jsonify(data)

@app.route("/funding-rate")
def funding_rate():
    data = fetch_funding_rate()
    return jsonify(data)

@app.route("/liquidations")
def liquidations():
    data = fetch_liquidation_data()
    return jsonify(data)

# _______________________________ Test Binance API ________________________________
@app.route("/price")
def price():
    data = get_price_from_binance()
    return jsonify(data)

@app.route("/month-price")
def month_price():
    data = get_month_prices()
    return jsonify(data)

@app.route("/four-hour-price")
def four_hour_price():
    data = get_four_hour_prices()
    return jsonify(data)

@app.route("/day-price")
def day_price():
    data = get_day_prices()
    return jsonify(data)

@app.route("/hour-price")
def hour_price():
    data = get_hour_prices()
    return jsonify(data)

@app.route("/week-price")
def week_price():
    data = get_week_prices()
    return jsonify(data)

@app.route("/predict-next-price")
def predict_next_price_route():
    try:
        # Fetch the data
        prices = get_month_prices()
        print("Fetched Prices:", prices[:5])  # Debug: Print the first 5 entries

        # Preprocess the data
        df, scaler = preprocess_data(prices)
        print("Preprocessed DataFrame Head:\n", df.head())  # Debug: Check DataFrame
        print("Scaler Min:", scaler.data_min_, "Scaler Max:", scaler.data_max_)  # Debug: Check scaler

        # Build and train the model
        model = build_and_train_model(df)
        print("Model Summary:")
        model.summary()  # Debug: Print model architecture

        # Make a prediction
        next_price = predict_next_price(model, df, scaler)
        print("Predicted Next Price:", next_price)  # Debug: Check prediction

        # Check if next_price is a valid array
        if next_price is not None and len(next_price) > 0:
            # Convert the prediction to a standard Python float
            predicted_price = float(next_price[0])
        else:
            raise ValueError("Prediction failed or returned an empty result.")

        # Return the prediction as JSON
        return jsonify({"predicted_next_price": predicted_price})
    except Exception as e:
        print("Error occurred:", str(e))  # Debug: Log error
        return jsonify({"error": str(e)}), 500

@app.route("/predict-combined")
def predict_combined_models():
    try:
        # Fetch the data for all models
        hour_prices = get_hour_prices()
        
        # Create DataFrame for poly and lasso models
        df = pd.DataFrame([{
            'close': p['close'],
            'high': p['high'],
            'low': p['low'],
            'open': p['open'],
            'volume': p['volume']
        } for p in hour_prices])
        
        # Number of future points to predict
        n_future = 12  # For example, next 12 time periods
        
        # Generate predictions for multiple future points
        lstm_predictions = []
        poly_predictions = []
        lasso_predictions = []
        
        # Preprocess data for LSTM model
        lstm_df, scaler = preprocess_data(hour_prices)
        lstm_model = build_and_train_model(lstm_df)
        
        # Build polynomial and lasso models
        poly_model = build_and_train_poly_model(df)
        lasso_model = build_and_train_lasso_model(df)
        
        current_df = df.copy()
        current_lstm_df = lstm_df.copy()
        
        for i in range(n_future):
            # Get predictions for each model
            lstm_pred = predict_next_price(lstm_model, current_lstm_df, scaler)
            poly_pred = predict_next_poly_price(poly_model, current_df)
            lasso_pred = predict_next_lasso_price(lasso_model, current_df)
            
            # Append predictions
            lstm_predictions.append(float(lstm_pred[0]))
            poly_predictions.append(float(poly_pred))
            lasso_predictions.append(float(lasso_pred))
            
            # Update dataframes with new predictions for next iteration
            new_row = pd.DataFrame([{
                'close': float(lstm_pred[0]),
                'high': float(lstm_pred[0]),  # Simplified
                'low': float(lstm_pred[0]),   # Simplified
                'open': float(lstm_pred[0]),  # Simplified
                'volume': df['volume'].mean() # Simplified
            }])
            current_df = pd.concat([current_df[1:], new_row]).reset_index(drop=True)
            
            # Update LSTM dataframe similarly
            current_lstm_df = pd.concat([current_lstm_df[1:], new_row]).reset_index(drop=True)
        
        # Get current price for comparison
        current_price = get_price_from_binance()
        
        # Prepare response with prediction series
        response = {
            "current_price": current_price,
            "predictions": {
                "polynomial": poly_predictions,
                "lasso": lasso_predictions,
                "lstm": lstm_predictions
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/prediction-history")
def get_prediction_history():
    try:
        # Get the limit parameter from query string, default to 100
        limit = int(request.args.get('limit', 100))
        history = prediction_logger.get_recent_predictions(limit)
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/model-accuracy")
def get_model_accuracy():
    try:
        accuracy_report = accuracy_tracker.get_accuracy_report()
        return jsonify(accuracy_report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# _______________________________________________________________________________

# Add after app initialization
prediction_logger = PredictionLogger()
accuracy_tracker = ModelAccuracyTracker()

# Add this new function for background logging
def background_prediction_logger():
    previous_prediction = None
    previous_price = None
    
    while True:
        try:
            # Get predictions
            with app.app_context():
                response = predict_combined_models()
                current_prediction_data = response.get_json()
                current_price = current_prediction_data["current_price"]
                
                # Update accuracy metrics if we have a previous prediction
                if previous_prediction and previous_price:
                    accuracy_tracker.update_accuracy(
                        previous_prediction,
                        current_price,
                        previous_price
                    )
                
                # Log new prediction
                prediction_logger.log_prediction(current_prediction_data)
                
                # Store current prediction for next comparison
                previous_prediction = current_prediction_data
                previous_price = current_price
                
            # Wait for 5 minutes
            time.sleep(300)  # 300 seconds = 5 minutes
            
        except Exception as e:
            print(f"Error in background logger: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying if there's an error

if __name__ == "__main__":
    # Start the background logging thread
    logging_thread = threading.Thread(target=background_prediction_logger, daemon=True)
    logging_thread.start()
    
    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=5001)
