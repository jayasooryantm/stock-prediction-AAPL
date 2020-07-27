import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


df = pd.read_csv('/content/AAPL.csv')

stockClosed = df['close']


scaler = MinMaxScaler(feature_range=(0,1))
stockClosed = scaler.fit_transform(np.array(stockClosed).reshape(-1,1))

training_size = int(len(stockClosed)*0.65)
testing_size = len(stockClosed)-training_size
train_data, test_data = stockClosed[0:training_size,:],
                        stockClosed[training_size:len(stockClosed),:]


def dataset_gen(data, time_step):
  dataX, dataY =[],[]
  for i in range(len(data)-time_step-1):
    dataX.append(data[i:(i+time_step),0])
    dataY.append(data[i+time_step,0])
  return np.array(dataX), np.array(dataY)


x_train, y_train = dataset_gen(train_data,100)
x_test, y_test = dataset_gen(test_data,100)


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)




lstm_model = Sequential()
lstm_model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
lstm_model.add(LSTM(50,return_sequences=True))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')


lstm_model.fit(x=x_train,y=y_train, 
               validation_data=(x_test,y_test), epochs=100, 
               batch_size=64, verbose=1)



train_predict = lstm_model.predict(x_train)
test_predict = lstm_model.predict(x_test)


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)



print('Train RMSE: ',np.sqrt(mean_squared_error(y_true=y_train,
                                                y_pred=train_predict)))
print('Test RMSE: ',np.sqrt(mean_squared_error(y_true=y_test,
                                               y_pred=test_predict)))



