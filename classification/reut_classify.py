from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

reut = keras.datasets.reuters

(x_train, y_train), (x_test, y_test) = reut.load_data(path="reuters.npz",
						 num_words=8000,
						 skip_top=0,
						 maxlen=None,
						 test_split=0.2,
						 seed=113,
						 start_char=1,
						 oov_char=2,
						 index_from=3)

print("train data:", x_train.shape)
print("train label:", y_train.shape)

def vectorize_seq(sequences, dimension=8000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

train_data = vectorize_seq(x_train)
test_data = vectorize_seq(x_test)

print("train_data", train_data.shape)
print("test_data", test_data.shape)

train_label = np.asarray(y_train).astype('float32')
test_label = np.asarray(y_test).astype('float32')

print("train+label", train_label.shape)
print("test_label", test_label.shape)

x_val = train_data[:8000]
partial_x_train = train_data[8000:]
y_val = train_label[:8000]
partial_y_train = train_label[8000:]

model = keras.models.Sequential()
model.add(layers.Dense(16, activation=tf.nn.relu))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation=tf.nn.relu))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation=tf.nn.sigmoid))

pochs = 10
size = 512

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(partial_x_train, partial_y_train, epochs=pochs, batch_size=size, validation_data=(x_val, y_val))

results = model.evaluate(test_data, test_label)
print("Test Loss and Accuracy")
print("results ", results)
history_dict = history.history
history_dict.keys()

plt.clf()
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, (len(history_dict['loss']) + 1))
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.predict(test_data)
