# Backend Model Explanation
The model is designed to predict the next closing price of bitcoin based on historical data. It uses a Long Short-Term Memory (LSTM) neural network, which is well-suited for time series prediction tasks due to its ability to capture temporal dependencies.

## Key Components

### 1. Data Preprocessing:
- **Scaling**: 
  - The data is first scaled using `MinMaxScaler` to normalize the features (close, high, low, open, volume) to a range between 0 and 1. 
  - This is crucial for neural networks as it helps in faster convergence and better performance.
- **Sequence Creation**:
  - The data is then transformed into sequences of a fixed length (30 in this case).
  - Each sequence is used as an input to the model, and the model predicts the next closing price.

### 2. Model Architecture:
- **LSTM Layers**:
  - The model consists of two LSTM layers. LSTMs are a type of recurrent neural network (RNN) that can learn long-term dependencies, which is useful for time series data.
  - The first LSTM layer returns sequences, which means it outputs a sequence of predictions for each time step in the input sequence.
  - The second LSTM layer processes these sequences and outputs a single prediction.
- **Dense Layer**:
  - A dense (fully connected) layer is used to produce the final output, which is the predicted closing price.

### 3. Training:
- **Loss Function**:
  - The model is trained using the Mean Squared Error (MSE) loss function, which is standard for regression tasks.
- **Optimizer**:
  - The optimizer used is Adam, which is an adaptive learning rate optimization algorithm that is efficient and widely used in deep learning.

### 4. Prediction:
- After training, the model can predict the next closing price based on the last 30 data points.
- The prediction is then inverse transformed to convert it back to the original scale using the `MinMaxScaler`.

## How It Works

1. **Input Data**:
   - The model takes in sequences of 30 data points, each containing the features close, high, low, open, and volume.
2. **LSTM Processing**:
   - The LSTM layers process these sequences to learn patterns and dependencies in the data.
   - The first LSTM layer outputs a sequence of predictions, which are then processed by the second LSTM layer to produce a single prediction.
3. **Output**:
   - The dense layer outputs the predicted closing price for the next time step.
4. **Inverse Transformation**:
   - The predicted value is transformed back to the original scale using the inverse of the scaling operation applied during preprocessing.

## Why LSTM?
LSTMs are chosen for this task because they are capable of learning from sequences of data and can remember information for long periods, which is essential for time series prediction. They are particularly effective in capturing the temporal dependencies and trends in financial data.

By following this process, the model aims to provide accurate predictions of future closing prices based on historical data.
