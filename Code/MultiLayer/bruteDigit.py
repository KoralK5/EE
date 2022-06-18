import numpy as np
import pandas as pd
import tensorflow as tf
import os
from time import time
from tensorflow.keras import layers
from keras.datasets.mnist import load_data
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
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
min_layers = 1
max_layers = 3
accuracies = []
layer_cnt = []
neuron_cnt = []

for i in range(min_layers, max_layers+1):
    for j in range(mn, mx+1, inc):
        layer_cnt.append(i)
        neuron_cnt.append(j)

start_tot = time()
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

for neurons in range(mn, mx+1, inc):
    print('neurons:', neurons)

    model = tf.keras.Sequential([
        layers.Dense(neurons, input_shape=(X.shape[1],), activation='relu'),
        layers.Dense(neurons, activation='relu'),
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

for neurons in range(mn, mx+1, inc):
    print('neurons:', neurons)

    model = tf.keras.Sequential([
        layers.Dense(neurons, input_shape=(X.shape[1],), activation='relu'),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(neurons, activation='relu'),
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
end_tot = time()

runtime = end_tot - start_tot

x = np.asarray(neuron_cnt)
y = np.asarray(layer_cnt)
z = np.asarray(accuracies)

print('neurons', x)
print('layers', y)
print('accuracies', z)
print('runtime', runtime)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, cmap='RdYlGn')
ax.set_title('Accuracy Across Different Hidden Neuron Combinations')
ax.set_xlabel('Hidden Neurons Per Layer')
ax.set_ylabel('Layer Count')
ax.set_zlabel('Accuracy')
plt.savefig(os.getcwd() + '\\Code\\Multilayer\\Graph\\Brute\\brute2.png')
plt.show()
