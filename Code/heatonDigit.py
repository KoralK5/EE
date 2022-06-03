import numpy as np
from keras.datasets.mnist import load_data

from Models.MLP import *

def get_data():
    (X, Y), (Xtest, Ytest) = load_data()
    dimensions = X.shape[1] * X.shape[2]

    X = X.reshape((X.shape[0], dimensions)) / 255.0
    Xtest = Xtest.reshape((Xtest.shape[0], dimensions)) / 255.0

    (X, Y), (Xval, Yval) = split(X, Y, train_frac=0.9)

    Y = to_categorical(Y)

    return X, Y, Xtest, Ytest, Xval, Yval

def heaton(I, O):
    return round((2/3)*I + O)

X, Y, Xtest, Ytest, Xval, Yval = get_data()

hidden_cnt = heaton(X.shape[1], Y.shape[1])

layers = [
    InputLayer(X.shape[1]),
    Layer(hidden_cnt, ReLU()),
    Layer(Y.shape[1], Softmax())
]

model = NeuralNetwork(layers)
model.compile(loss_function=SquaredLoss(), metric=accuracy)

history = model.fit(X, Y, Xval, Yval, learning_rate=0.01, n_epochs=10, batch_size=32)

model.print_predict(Xtest, Ytest)
plot_loss_and_accuracy(history)
