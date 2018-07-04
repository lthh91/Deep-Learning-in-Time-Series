# Deep-Learning-in-Time-Series
Using a deep Neural Net to predict the USD/EUR exchange rate

This project predicts Univariate time series using Neural Networks.

The idea is to use one observation as the output, whose input data is the n previous observations. The variable look_back is use to determine how many observations to be included in one input, i.e. the size of the X matrix in the Neural Net.

After creating input and output data, the process is conducted normally as training and testing a neural net.

The project uses keras with Tensorflow backend to process the Neural net.
