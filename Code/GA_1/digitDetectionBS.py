import numpy as np
from keras.datasets.mnist import load_data

from Models.NN import *

def fetch_data():
    (X, Y), (Xtest, Ytest) = load_data()
    dimensions = X.shape[1] * X.shape[2]

    X = X.reshape((X.shape[0], dimensions)) / 255.0
    Xtest = Xtest.reshape((Xtest.shape[0], dimensions)) / 255.0

    (X, Y), (Xval, Yval) = split(X, Y, train_frac=0.9)
    Y = to_categorical(Y)

    return X, Y, Xval, Yval

def binary_search(X, Y, Xval, Yval, mi, ma):
    neurons = []
    losses = [float('inf')]
    model = None
    inc = True
    while(mi <= ma):
        mid = (mi + ma) // 2

        layers = [
            InputLayer(X.shape[1]),
            Layer(mid, ReLU()),
            Layer(10, Softmax())
        ]

        model = NeuralNetwork(layers)
        model.compile(loss_function=SquaredLoss(), metric=accuracy)

        history = model.fit(X, Y, Xval, Yval, learning_rate=0.01, n_epochs=1, batch_size=32)

        loss = history['Val_loss'][-1]

        if (loss < losses[-1]) ^ (inc):
            ma = mid - 1
            inc = False
        else:
            mi = mid + 1
            inc = True
        
        print(loss, '->' if inc else '<-', mid, 'neurons')
        losses.append(loss)
        neurons.append(mid)

    return model, losses, neurons

X, Y, Xval, Yval = fetch_data()
model, losses, neurons = binary_search(X, Y, Xval, Yval, 1, 200)

plt.scatter(neurons, losses)
plt.show()
