import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
from tensorflow.keras import layers
from keras.datasets.fashion_mnist import load_data
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

def fetch_data():
    (X, Y), (Xtest, Ytest) = load_data()
    dimensions = X.shape[1] * X.shape[2]

    X = X.reshape((X.shape[0], dimensions)) / 255.0
    Xtest = Xtest.reshape((Xtest.shape[0], dimensions)) / 255.0

    Y = tf.keras.utils.to_categorical(Y)
    Ytest = tf.keras.utils.to_categorical(Ytest)

    X, Xval, Y, Yval = train_test_split(X, Y, test_size=0.1, random_state=42)

    return X, Y, Xtest, Ytest, Xval, Yval

def timothy(i, o):
    return int(np.sqrt(i*o))

X, Y, Xtest, Ytest, Xval, Yval = fetch_data()

print('X-dims:', X.shape)
print('Y-dims:', Y.shape)

neurons = timothy(X.shape[1], Y.shape[1])
print('neurons:', neurons)

model = tf.keras.Sequential([
    layers.Dense(neurons, input_shape=(X.shape[1],), activation='relu'),
    layers.Dense(Y.shape[1], activation='softmax')
])

model.compile(optimizer='SGD', loss='mse', metrics='accuracy')

start = time()
history = model.fit(X, Y, validation_data=(Xval, Yval), epochs=10)
end = time()

runtime = end - start

print(runtime, 'seconds')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

realY = np.argmax(Ytest.copy(), axis=1)

start_t = time()
predY = np.argmax(model.predict(Xtest), axis=1)
end_t = time()

runtime_t = end_t - start_t

print(runtime_t, 'seconds')

cf_matrix = confusion_matrix(realY, predY)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('\nActual Values')

ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names, rotation=0)

plt.show()