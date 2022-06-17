import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
from tensorflow.keras import layers
from keras.datasets.mnist import load_data
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

def fetch_data():
    (X, Y), (Xtest, Ytest) = load_data()
    dimensions = X.shape[1] * X.shape[2]

    X = X.reshape((X.shape[0], dimensions)) / 255.0
    Xtest = Xtest.reshape((Xtest.shape[0], dimensions)) / 255.0

    Y = tf.keras.utils.to_categorical(Y)
    Ytest = tf.keras.utils.to_categorical(Ytest)

    X, Xval, Y, Yval = train_test_split(X, Y, test_size=0.1, random_state=42)

    return X, Y, Xtest, Ytest, Xval, Yval

def create_model(neurons):
    all_layers = [layers.Dense(neurons[0], input_shape=(X.shape[1],), activation='relu')]
    for n in neurons[1:]:
        all_layers.append(layers.Dense(n, activation='relu'))
    all_layers.append(layers.Dense(Y.shape[1], activation='softmax'))

    model = tf.keras.Sequential(all_layers)
    model.compile(optimizer='SGD', loss='mse', metrics='accuracy')
    return model

def timothy(i, o):
    return int(np.sqrt(i*o))

X, Y, Xtest, Ytest, Xval, Yval = fetch_data()

print('X-dims:', X.shape)
print('Y-dims:', Y.shape)

I = X.shape[1]
neurons = []
times = []
for i in range(5):
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    neurons.append(timothy(I, Y.shape[1]))
    I = neurons[-1]
    model = create_model(neurons)

    print(neurons)

    start = time()
    history = model.fit(X, Y, validation_data=(Xval, Yval), epochs=10)
    end = time()

    runtime = end - start
    times.append(runtime)

print('neurons', neurons)
print('layers', list(range(1, 6)))
print('runtime', times)
