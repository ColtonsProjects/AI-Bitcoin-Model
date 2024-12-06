import os
import requests
import time
import datetime
from model import (
    preprocess_data, 
    build_and_train_model, 
    predict_next_price,
    build_and_train_xgboost,
    predict_next_price_xgboost,
    build_and_train_prophet,
    predict_next_price_prophet
)
from flask import Flask, jsonify
import numpy as np
from dotenv import load_dotenv
from flask_cors import CORS


load_dotenv()

app = Flask(__name__)
CORS(app)

# ______________________________ Coinalyze API ______________________________

def fetch_open_interest():
    api_key = os.getenv('COINALYZE_API_KEY')
    url = "https://api.coinalyze.net/v1/open-interest"
    params = {
        'symbols': 'BTCUSD_PERP.A', 
        'convert_to_usd': 'true'
    }
    headers = {
        'api_key': f'{api_key}'
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch data", "status_code": response.status_code}

def fetch_funding_rate():
    api_key = os.getenv('COINALYZE_API_KEY')
    url = "https://api.coinalyze.net/v1/funding-rate"
    params = {
        'symbols': 'BTCUSD_PERP.A' 
    }
    headers = {
        'api_key': f'{api_key}'
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch data", "status_code": response.status_code}
    
def fetch_liquidation_data():
    api_key = os.getenv('COINALYZE_API_KEY')
    url = "https://api.coinalyze.net/v1/liquidation-history"
    symbol = 'BTCUSD_PERP.A'  # Ensure this matches Coinalyze's symbol for the desired contract

    # Define the time interval and range
    interval = '1hour'  # Options: '1min', '5min', '15min', '30min', '1hour', '2hour', '4hour', '6hour', '12hour', 'daily'
    current_time = int(time.time())
    one_week_ago = current_time - 7 * 24 * 60 * 60  # 7 days ago

    params = {
        'symbols': symbol,
        'interval': interval,
        'from': one_week_ago,
        'to': current_time,
        'convert_to_usd': 'true'  # Convert values to USD
    }

    headers = {
        'api_key': api_key
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error occurred: {http_err}", "status_code": response.status_code}
    except Exception as err:
        return {"error": f"An error occurred: {err}"}
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
    start_time = end_time - (60 * 60 * 1000)  # Past hour in milliseconds

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
        print("Fetched Prices:", prices[:5])

        # Preprocess the data
        df, scaler = preprocess_data(prices)
        print("Preprocessed DataFrame Head:\n", df.head())

        predictions = {}

        # 1. Original LSTM prediction
        lstm_model = build_and_train_model(df)
        lstm_prediction = predict_next_price(lstm_model, df, scaler)
        if lstm_prediction is not None and len(lstm_prediction) > 0:
            predictions['lstm'] = float(lstm_prediction[0])

        # 2. XGBoost prediction
        xgb_model = build_and_train_xgboost(df)
        xgb_prediction = predict_next_price_xgboost(xgb_model, df, scaler)
        if xgb_prediction is not None and len(xgb_prediction) > 0:
            predictions['xgboost'] = float(xgb_prediction[0])

        # 3. Prophet prediction
        prophet_model = build_and_train_prophet(df)
        prophet_prediction = predict_next_price_prophet(prophet_model, df)
        if prophet_prediction is not None:
            predictions['prophet'] = float(prophet_prediction)

        # Calculate average prediction
        valid_predictions = [p for p in predictions.values() if p is not None]
        if valid_predictions:
            predictions['average'] = sum(valid_predictions) / len(valid_predictions)

        if not predictions:
            raise ValueError("All predictions failed or returned empty results.")

        # Return all predictions as JSON
        return jsonify({
            "predictions": predictions,
            "timestamp": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({"error": str(e)}), 500

# _______________________________________________________________________________
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
