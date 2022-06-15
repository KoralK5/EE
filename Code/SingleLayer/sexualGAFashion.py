import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
from random import randint
from tensorflow.keras import layers
from keras.datasets.fashion_mnist import load_data
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

def timothy(i, o):
    return int(np.sqrt(i*o))

def heaton(i, o):
    return int((2/3)*i + o)

def create_model(n):
    model = tf.keras.Sequential([
        layers.Dense(n, input_shape=(X.shape[1],), activation='relu'),
        layers.Dense(Y.shape[1], activation='softmax')
    ])
    model.compile(optimizer='SGD', loss='mse', metrics='accuracy')
    return model

def mutate(n, rate=100, ub=533, lb=1):
    return randint(max(n-rate, lb), min(n+rate, ub))

def create_batch(parent, size=5):
    topology = set()
    topology.add(parent)
    while len(topology) < size:
        neurons = mutate(parent)
        topology.add(neurons)
    return list(topology)

def fitness(model, Xtest, Ytest):
    realY = np.argmax(Ytest.copy(), axis=1)
    predY = np.argmax(model.predict(Xtest), axis=1)
    accuracy = np.sum(realY == predY) / len(Ytest)
    return accuracy

def selection(models, Xtest, Ytest):
    best_acc = 0
    best_idx = 0
    sec_acc = 0
    sec_idx = 0
    for idx in range(len(models)):
        cur_model = models[idx]
        cur_acc = fitness(cur_model, Xtest, Ytest)
        if cur_acc >= best_acc:
            sec_acc = best_acc
            sec_idx = best_idx
            best_acc = cur_acc
            best_idx = idx
    return best_idx, best_acc, sec_idx, sec_acc

X, Y, Xtest, Ytest, Xval, Yval = fetch_data()

print('X-dims:', X.shape)
print('Y-dims:', Y.shape)

neurons = timothy(X.shape[1], Y.shape[1])
bound = heaton(X.shape[1], Y.shape[1])
print('neurons:', neurons)

start = time()
trained_models = []
best_histories = []
histories = []
training_hist = []
parent = neurons
parents = [parent]
for i in range(10):
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    histories.append([])
    models = []
    topology = create_batch(parent)
    for top in topology:
        cur_model = create_model(top)
        cur_history = cur_model.fit(X, Y, validation_data=(Xval, Yval), epochs=10)
        models.append(cur_model)
        histories[-1].append(cur_history)

    idx_1, parent_acc_1, idx_2, parent_acc_2 = selection(models, Xtest, Ytest)
    parent = round((topology[idx_1] + topology[idx_2]) / 2)
    trained_models.append(models[idx_1])
    training_hist.append(histories[-1][idx_1])
    best_histories.append(parent_acc_1)
    parents.append(parent)

end = time()
runtime = end - start

print(runtime, 'seconds')
print('parents', parents)

model = trained_models[-1]
history = training_hist[-1]

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

plt.bar(range(1, len(parents)+1), parents)
plt.title('The Evolution of Hidden Neuron Count')
plt.ylabel('Hidden Neurons')
plt.xlabel('Generation')
plt.show()

plt.bar(range(1, len(best_histories)+1), best_histories)
plt.yscale('log')
plt.title('The Accuracy of the Best Model in each Batch')
plt.ylabel('Accuracy')
plt.xlabel('Generation')
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
