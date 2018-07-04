import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
import math
import matplotlib.pyplot as plt

#dataframe = pd.read_excel('DAT_XLSX_EURUSD_M1_201806.xlsx', usecols=[2])
dataframe = pd.read_csv('EURUSD-1440.csv',names=['Date','Hour','Open','High','Low','Close','Volume'], index_col='Date')
dataframe.index = pd.to_datetime(dataframe.index)
data = dataframe['Close']
dataset = data.values
dataset = dataset.astype('float32')

train_size = int(len(dataset)*0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

look_back = 150
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

def create_network(look_back, layers):
    model = Sequential()
    model.add(Dense(layers[0], input_dim=look_back, activation='relu'))
    for i in range(1,len(layers)):
        model.add(Dense(layers[i], activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

early_stopping = [EarlyStopping(patience=3)]

model = create_network(look_back, [8, 30, 30, 30, 30, 30, 30, 8])
model.fit(trainX, trainY, epochs=500, batch_size=2, verbose=2, callbacks=early_stopping)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' %(trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' %(testScore, math.sqrt(testScore)))

# Serialize model to json
model_json = model.to_json()
with open("model3.json", "w") as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5
model.save_weights('model3.h5')
print("Saved model to disk")
