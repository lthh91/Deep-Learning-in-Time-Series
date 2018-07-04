import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import math
import matplotlib.pyplot as plt

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
model = loaded_model

dataframe = pd.read_csv('EURUSD-1440.csv',names=['Date','Hour','Open','High','Low','Close','Volume'], index_col='Date')
data = dataframe['Close']
dataset = data.values
dataset = dataset.astype('float32')

look_back = 150

predict_length = 200
#Generate predictions:
X_predict = np.array([dataset[-look_back:]])
Y_predict = []
index = pd.to_datetime(dataframe.index)
print(index)
for i in range(predict_length):
    new_Y = model.predict(X_predict)
    Y_predict.append(np.asscalar(new_Y))
    X_predict = np.append(X_predict, new_Y, 1)
    X_predict = X_predict[:,-look_back:]
    index = np.append(index, index[-1]+pd.to_timedelta(1,unit='d'))

# index = pd.to_datetime(index)
# print(index)

Y_Plot = np.empty_like(dataset)
Y_Plot[:] = np.nan
Y_Plot[-1] = dataset[-1]
Y_Plot = np.append(Y_Plot, Y_predict)
Data_Plot = np.empty_like(Y_predict)
Data_Plot[:] = np.nan
Data_Plot = np.append(dataset, Data_Plot)
df = pd.DataFrame({'Real data':Data_Plot, 'Predicted':Y_Plot}, index=index)
df.to_csv('predict1.csv')
df_plot = df.iloc[-550:,:]
# print(df)
df_plot.plot()
plt.savefig('predict1.png')
plt.show()
