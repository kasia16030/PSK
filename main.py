import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf


file_path = 'us26_data/1.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

from_node = lines[0].strip()
to_node = lines[1].strip()
traffic_values = [float(value.strip()) for value in lines[2:]]

data = np.reshape(traffic_values, (len(traffic_values), 1))
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

window_size = 100
X, y = [], []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

train_split = 0.8
validation_split = 0.1
test_split = 0.1

train_split_index = int(len(scaled_data) * train_split)
validation_split_index = int(len(scaled_data) * (train_split + validation_split))

X_train = X[:train_split_index]
Y_train = y[:train_split_index]
X_validation = X[train_split_index:validation_split_index]
Y_validation = y[train_split_index:validation_split_index]
X_test = X[validation_split_index:]
Y_test = y[validation_split_index:]

print("Rozmiar zbioru treningowego:", len(X_train))
print("Rozmiar zbioru walidacyjnego:", len(X_validation))
print("Rozmiar zbioru testowego:", len(X_test))

time_train = range(len(Y_train))
time_validation = range(len(Y_train), len(Y_train) + len(Y_validation))
time_test = range(len(Y_train) + len(Y_validation), len(Y_train) + len(Y_validation) + len(Y_test))

plt.figure(figsize=(14, 5))
plt.plot(time_train, Y_train)
plt.plot(time_validation,Y_validation)
plt.plot(time_test, Y_test)
plt.title('Data')
plt.legend(['Train', 'Val', 'Test'])
plt.show()

model_lstm = tf.keras.models.Sequential()
model_lstm.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(tf.keras.layers.Dropout(0.1))
model_lstm.add(tf.keras.layers.LSTM(units=50))
model_lstm.add(tf.keras.layers.Dropout(0.1))
model_lstm.add(tf.keras.layers.Dense(units=20, activation='tanh'))
model_lstm.add(tf.keras.layers.Dropout(0.1))
model_lstm.add(tf.keras.layers.Dense(units=1))

model_lstm.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mae'])
model_lstm.summary()

history_lstm = model_lstm.fit(X_train, Y_train, epochs=30, batch_size=128, validation_split=0.2)

model_lstm.evaluate(X_test, Y_test)

plt.plot(history_lstm.history['loss'])
plt.plot(history_lstm.history['val_loss'])
plt.title('LSTM loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = []
num_predictions = 1000

current_window = X_test[0].reshape(1, -1, 1)

for i in range(num_predictions):
    predict_val = model_lstm.predict(current_window)
    predictions.append(predict_val[0, 0])
    current_window = np.concatenate((current_window[:, 1:, :], predict_val.reshape(1, 1, 1)), axis=1)

y_test_actual = scaler.inverse_transform(np.array(Y_test).reshape(-1, 1))
predictions_actual = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

plt.figure(figsize=(15, 6))
plt.plot(y_test_actual[:num_predictions], label='actual', color='blue')
plt.plot(predictions_actual, label='predicted', color='red')
plt.title('LSTM prediction - 1000 values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

