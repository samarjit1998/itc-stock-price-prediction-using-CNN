
#import libary
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import Adam


# command line argument fetching data file
dataFilePath = sys.argv[1]

# load dataset
raw_dataset = pd.read_csv(dataFilePath)


# explore dataset or understand the data
print(raw_dataset.head())
print(raw_dataset.corr())
# we want to predict the 'close'

# feature selection : we only consider the close feature
close = raw_dataset["Close"].to_numpy()
date = raw_dataset["Date"].to_numpy()
print("Close : ", close)

# data vizualization
x_dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in date.tolist()]
y_close = close.tolist()
plt.plot(x_dates, y_close)
plt.xlabel("dates")
plt.ylabel("close")
# plt.show()


# feature selection
# creating the features and target with defined window size
feature = []
target = []
window_size = 7
for i in range(window_size+1, len(y_close)):
    target.append(y_close[i])
    datarow = []
    for j in range(i-window_size, i):
        datarow.append(y_close[j])
    feature.append(datarow)

print("features : ", feature[0:5])
print("targets : ", target[0:5])

# data processing
# splitting train test data
train_perct = 80
dividor = int((len(y_close)*train_perct)/100)
x_train = np.array(feature[:dividor])
y_train = np.array(target[:dividor])
x_test = np.array(feature[dividor:])
y_test = np.array(target[dividor:])

# proper shapping to fit io to CNN
x_train = x_train.reshape((-1, 7, 1))
x_test = x_test.reshape((-1, 7, 1))
print("X_train shape : ", x_train.shape)
print("X_test shape : ", x_test.shape)


# creating the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3,
                 activation='relu', input_shape=(7, 1)))
# model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))
opt = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=opt, metrics=['mse'])

# tranning
batch_size = 64
epochs = 5
verbose = True
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_test, y_test), verbose=verbose)


# testing the model with test data and evaluting
ground_truth = y_test
predicted = model.predict(x_test).tolist()
size_of_groundTruth = ground_truth.size
x_axis_data = []
for i in range(size_of_groundTruth):
    x_axis_data.append(x_dates[dividor+i])

plt.plot(x_axis_data, predicted, '--',
         label="Predicion", color="red")
plt.plot(x_axis_data, ground_truth.tolist(),
         label="Ground Truth", color="blue")
plt.xlabel("dates")
plt.ylabel("close")
plt.legend()
plt.show()


def errorCalculation(groundTruth, predicted):
    square_errors = 0
    size = 0
    for i in range(groundTruth.size):
        error = groundTruth[i]-predicted[i]
        square_error = error * error
        square_errors = square_errors + square_error
        size = size + 1
    return square_errors / size


print("MSE of Prediction with Ground Truth : ",
      errorCalculation(ground_truth, predicted))
