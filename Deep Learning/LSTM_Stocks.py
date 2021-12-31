import datetime
import pandas as pd
from pandas_datareader import data as web
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.pylab import rcParams
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn import *

#Load Data

ticker="AMZN"

start = datetime.datetime(2018,1,1)
end = datetime.datetime(2021,1,1)

data = web.DataReader(ticker,'yahoo',start, end)
data.head()

# Normalize Training Dataset

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# train on 90 day data points
learn_seq = 90

x_train = []
y_train = []

for x in range(learn_seq,len(scaled_data)):
    x_train.append(scaled_data[x-learn_seq:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))



# LSTM Model: 3-->1 layers (experiment for best results)

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))
          
#model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

md=model.fit(x_train, y_train, epochs=30, batch_size=32)        
     


# Plot Model Loss(epoch):
plt.plot(md.history['loss'])
plt.title('')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# Test & Predict

test_start = datetime.datetime(2021,1,1)
test_end = datetime.datetime(2021,12,10)

test_data = web.DataReader(ticker,'yahoo',test_start, test_end)
actual_prices = test_data['Close'].values

# Model input params
model_inputs = test_data['Close'].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)


x_test = []

for x in range(learn_seq,len(model_inputs)):
    x_test.append(model_inputs[x-learn_seq:x, 0])


x_test= np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot Predicted vs Actual Price

style.use('ggplot')
plt.plot(actual_prices[learn_seq:], color="green", label="Actual Price")
plt.plot(predicted_prices, color="lime", label="Predicted Price")


plt.ylabel('Price ($)')
plt.xlabel('Days: 2021-3-1 to 2021-10-11')
plt.title('{} Historical Stock Price'.format(ticker))
plt.legend()
plt.show()