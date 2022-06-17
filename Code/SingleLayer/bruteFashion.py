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

X, Y, Xtest, Ytest, Xval, Yval = fetch_data()

print('X-dims:', X.shape)
print('Y-dims:', Y.shape)

mn = 10
mx = 550
inc = 10
accuracies = []
for neurons in range(mn, mx+1, inc):
    print('neurons:', neurons)

    model = tf.keras.Sequential([
        layers.Dense(neurons, input_shape=(X.shape[1],), activation='relu'),
        layers.Dense(Y.shape[1], activation='softmax')
    ])

    model.compile(optimizer='SGD', loss='mse', metrics='accuracy')

    start = time()
    history = model.fit(X, Y, validation_data=(Xval, Yval), epochs=10)
    end = time()

    realY = np.argmax(Ytest.copy(), axis=1)
    predY = np.argmax(model.predict(Xtest), axis=1)
    accuracy = np.sum(realY == predY) / len(Ytest)

    accuracies.append(accuracy)

print('accuracies', accuracies)

plt.plot(list(range(mn, mx+1, inc)), accuracies)
plt.title('Accuracy Across Different Hidden Layer Neuron Counts')
plt.ylabel('Validation Accuracy')
plt.xlabel('Hidden Neurons')
plt.show()