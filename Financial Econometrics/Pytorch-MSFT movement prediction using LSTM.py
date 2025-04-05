# Stock prediction using PyTorch with RSME variance & LSTM by K.Tomov

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

#last 3 years log-return data
end_date = datetime.today()
start_date = end_date - timedelta(days=3 * 365)
ticker_symbol = "MSFT"
msft_data = yf.download(ticker_symbol, start=start_date, end=end_date)
msft_data.rename(columns={'Volume': 'Volume1'}, inplace=True)
msft_data = msft_data[['Open', 'High', 'Low', 'Close', 'Volume1']]

#reset index
msft_data.reset_index(inplace=True)
msft_data['Date'] = pd.to_datetime(msft_data['Date'])
msft_data.set_index('Date', inplace=True)
print(msft_data.head())

#closing plot
plt.figure(figsize=(16, 8))
plt.plot(msft_data['Close'], label='Close Price History')
plt.title('MSFT Closing Price History (Past 3 Years)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()

#prep the data for LSTM

#sorting into dataframe
data = msft_data.sort_index(ascending=True)
new_data = data[['Close']].copy()
new_data['Date'] = new_data.index
new_data.reset_index(drop=True, inplace=True)

#train on past data - 252 days
train = new_data.iloc[:len(new_data) - 252].copy()
valid = new_data.iloc[len(new_data) - 252:].copy()

print('\nShape of training set:', train.shape)
print('Shape of validation set:', valid.shape)

#scaling close data
dataset = new_data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
training_data_len = len(train)

#initiating the LSTM sequence

look_back = 100  # past date(days) for predictive modeling

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

train_data = scaled_data[:training_data_len]
X_train, y_train = create_sequences(train_data, look_back)

# convert training data into PyTorch tensors
X_train_t = torch.from_numpy(X_train).type(torch.FloatTensor).unsqueeze(2)
y_train_t = torch.from_numpy(y_train).type(torch.FloatTensor)

#Pytorch LSTM Object

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))# Forward propagate LSTM
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Instantiate the model, define the loss function and the optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training the model

epochs = 30  # nb of times the model on training data set.
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs.squeeze(), y_train_t)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

#predictions of stock direction
test_data = scaled_data[training_data_len - look_back:]
X_test, _ = create_sequences(test_data, look_back)
X_test_t = torch.from_numpy(X_test).type(torch.FloatTensor).unsqueeze(2)

model.eval()
with torch.no_grad(): #gradient evaluator
    predictions = model(X_test_t).squeeze().numpy()

predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

#Calculate RMSE between predictions and actual validation close prices
valid_actual = valid['Close'].values
rmse = np.sqrt(np.mean((predictions.flatten() - valid_actual) ** 2))
print("\nPyTorch LSTM Model RMSE on validation set:")
print(rmse)

#ploting final results:

train_plot = new_data.iloc[:training_data_len]
valid_plot = new_data.iloc[training_data_len:].copy()
valid_plot['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.plot(train_plot['Close'], label='Training Data')
plt.plot(valid_plot['Close'], label='Validation Data')
plt.plot(valid_plot['Predictions'], label='Predictions')
plt.title('MSFT Stock Price Prediction using PyTorch LSTM')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()


