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
    def parse_datetime(row):
        return datetime.strptime(row['5 Minutes'], '%m/%d/%Y %H:%M')

    data['datetime'] = data.apply(parse_datetime, axis=1)
    data['hour'] = data['datetime'].dt.hour
    data['day'] = data['datetime'].dt.dayofweek
    data['month'] = data['datetime'].dt.month
    data['minute'] = data['datetime'].dt.minute
    data['timestamp'] = data['datetime'].astype(np.int64) // 10**9

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
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
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

input_size = X_train.shape[1]
hidden_size = 64
output_size = 1

model = TrafficRateNN(input_size, hidden_size, output_size)

# Train the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
def calculate_accuracy(predictions, true_values, tolerance=0.1):
    accuracy = (torch.abs(predictions - true_values) / true_values) < tolerance
    return accuracy.float().mean()

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    accuracy = calculate_accuracy(predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')
    print(f'Accuracy: {accuracy.item():.4f}')

# Function to predict packet rate given a date and timestamp
def predict_packet_rate(date_str):
    # Parse the input date string
    date_time = datetime.strptime(date_str, '%m/%d/%Y %H:%M')
    # Extract features
    hour = date_time.hour
    day = date_time.day
    month = date_time.month
    minute = date_time.minute
    # Create feature array
    features = np.array([[hour, day, month, minute]])
    print(features)
    # Normalize features
    features = scaler.transform(features)
    # Convert to PyTorch tensor
    features = torch.tensor(features, dtype=torch.float32)
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(features)
    return prediction.item()

# Example usage of the prediction function
date_str = '04/01/2016 15:00'
predicted_rate = predict_packet_rate(date_str)
print(f'Predicted packet rate for {date_str}: {predicted_rate}')
