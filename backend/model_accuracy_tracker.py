import json
from pathlib import Path
import numpy as np

class ModelAccuracyTracker:
    def __init__(self):
        self.base_dir = Path("prediction_logs")
        self.filepath = self.base_dir / "model_accuracy.json"
        self.metrics = {
            "polynomial": {
                "total_error": 0,
                "mean_error": 0,
                "predictions_count": 0,
                "mae": 0,  # Mean Absolute Error
                "mse": 0,  # Mean Squared Error
                "correct_direction": 0  # Number of times direction was correct
            },
            "lasso": {
                "total_error": 0,
                "mean_error": 0,
                "predictions_count": 0,
                "mae": 0,
                "mse": 0,
                "correct_direction": 0
            },
            "lstm": {
                "total_error": 0,
                "mean_error": 0,
                "predictions_count": 0,
                "mae": 0,
                "mse": 0,
                "correct_direction": 0
            }
        }
        self._load_metrics()

    def _load_metrics(self):
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                self.metrics = json.load(f)

    def _save_metrics(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def update_accuracy(self, previous_prediction, current_price, previous_price):
        if not previous_prediction:
            return

        for model_name, predictions in previous_prediction["predictions"].items():
            # Handle both single predictions and arrays of predictions
            predicted_price = predictions[0] if isinstance(predictions, list) else predictions
            
            # Calculate absolute error
            error = abs(current_price - predicted_price)
            squared_error = (current_price - predicted_price) ** 2
            
            # Update metrics
            self.metrics[model_name]["total_error"] += error
            self.metrics[model_name]["predictions_count"] += 1
            
            # Update mean error
            count = self.metrics[model_name]["predictions_count"]
            self.metrics[model_name]["mean_error"] = self.metrics[model_name]["total_error"] / count
            
            # Update MAE
            self.metrics[model_name]["mae"] = (
                (self.metrics[model_name]["mae"] * (count - 1) + error) / count
            )
            
            # Update MSE
            self.metrics[model_name]["mse"] = (
                (self.metrics[model_name]["mse"] * (count - 1) + squared_error) / count
            )
            
            # Check if direction was correct
            predicted_direction = predicted_price > previous_price
            actual_direction = current_price > previous_price
            if predicted_direction == actual_direction:
                self.metrics[model_name]["correct_direction"] += 1

        self._save_metrics()

    def get_accuracy_report(self):
        report = {}
        for model_name, metrics in self.metrics.items():
            predictions_count = metrics["predictions_count"]
            if predictions_count > 0:
                direction_accuracy = (metrics["correct_direction"] / predictions_count) * 100
                report[model_name] = {
                    "mean_error": round(metrics["mean_error"], 2),
                    "mae": round(metrics["mae"], 2),
                    "mse": round(metrics["mse"], 2),
                    "rmse": round(np.sqrt(metrics["mse"]), 2),
                    "direction_accuracy": round(direction_accuracy, 2),
                    "predictions_count": predictions_count
                }
        return report