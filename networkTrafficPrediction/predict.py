import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim

# Function to preprocess the data
def preprocess_data(data):
    data['datetime'] = pd.to_datetime(data['5 Minutes'], format='%m/%d/%Y %H:%M')
    data['hour'] = data['datetime'].dt.hour
    data['day'] = data['datetime'].dt.dayofweek
    data['month'] = data['datetime'].dt.month
    data['minute'] = data['datetime'].dt.minute
    data['timestamp'] = data['datetime'].astype('int64') // 10**9

    features = data[['hour', 'day', 'month', 'minute']].values
    target = data['Rate'].values

    return features, target

# Load the initial data
data_initial = pd.read_csv('packet_rate_data.csv')
X_initial, y_initial = preprocess_data(data_initial)

# Load additional data
data_additional = pd.read_csv('additional_packet_rate_data.csv')  # Replace with your additional data file
X_additional, y_additional = preprocess_data(data_additional)

# Combine initial and additional data
X_combined = np.concatenate((X_initial, X_additional), axis=0)
y_combined = np.concatenate((y_initial, y_additional), axis=0)

# Split the combined data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for batch
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the neural network
class TrafficRateNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TrafficRateNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Define RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_size = X_train.shape[2]
hidden_size = 64
output_size = 1
num_layers = 1

# Instantiate models
ffnn_model = TrafficRateNN(input_size, hidden_size, output_size)
rnn_model = RNNModel(input_size, hidden_size, output_size, num_layers)
lstm_model = LSTMModel(input_size, hidden_size, output_size, num_layers)
gru_model = GRUModel(input_size, hidden_size, output_size, num_layers)

models = {
    'FFNN': ffnn_model,
    'RNN': rnn_model,
    'LSTM': lstm_model,
    'GRU': gru_model
}

# Training function
def train_model(model, criterion, optimizer, X_train, y_train, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Function to calculate accuracy
def calculate_accuracy(predictions, true_values, tolerance=0.1):
    accuracy = (torch.abs(predictions - true_values) / true_values) < tolerance
    return accuracy.float().mean()

# Evaluate model
def evaluate_model(model, criterion, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        accuracy = calculate_accuracy(predictions, y_test)
        return test_loss.item(), accuracy.item()

# Train and evaluate each model
criterion = nn.MSELoss()
results = {}

for name, model in models.items():
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f'Training {name}...')
    train_model(model, criterion, optimizer, X_train, y_train)
    test_loss, accuracy = evaluate_model(model, criterion, X_test, y_test)
    results[name] = {'Test Loss': test_loss, 'Accuracy': accuracy}
    print(f'{name} - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

# Function to predict packet rate given a date and timestamp
def predict_packet_rate(date_str, model):
    # Parse the input date string
    date_time = datetime.strptime(date_str, '%m/%d/%Y %H:%M')
    # Extract features
    hour = date_time.hour
    day = date_time.weekday()
    month = date_time.month
    minute = date_time.minute
    # Create feature array
    features = np.array([[hour, day, month, minute]])
    # Normalize features
    features = scaler.transform(features)
    # Convert to PyTorch tensor
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for batch
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(features)
    return prediction.item()

# Example usage of the prediction function
date_str = '04/01/2016 15:00'
predicted_rate = predict_packet_rate(date_str, ffnn_model)
print(f'Predicted packet rate for {date_str} using FFNN: {predicted_rate}')

