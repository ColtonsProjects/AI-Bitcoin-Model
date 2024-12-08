import json
import datetime
import os
from pathlib import Path

class PredictionLogger:
    def __init__(self, filename="prediction_history.json"):
        self.base_dir = Path("prediction_logs")
        self.base_dir.mkdir(exist_ok=True)
        self.filepath = self.base_dir / filename
        self._initialize_file()

    def _initialize_file(self):
        if not self.filepath.exists():
            with open(self.filepath, 'w') as f:
                json.dump([], f)

    def log_prediction(self, prediction_data):
        try:
            # Read existing data
            with open(self.filepath, 'r') as f:
                predictions = json.load(f)

            # Add timestamp to prediction data
            timestamp = datetime.datetime.now().isoformat()
            prediction_entry = {
                "timestamp": timestamp,
                "current_price": prediction_data["current_price"],
                "predictions": prediction_data["predictions"],
            }

            # Append new prediction
            predictions.append(prediction_entry)

            # Write back to file
            with open(self.filepath, 'w') as f:
                json.dump(predictions, f, indent=2)

            return True
        except Exception as e:
            print(f"Error logging prediction: {str(e)}")
            return False

    def get_recent_predictions(self, limit=100):
        try:
            with open(self.filepath, 'r') as f:
                predictions = json.load(f)
            return predictions[-limit:]
        except Exception as e:
            print(f"Error reading predictions: {str(e)}")
            return []