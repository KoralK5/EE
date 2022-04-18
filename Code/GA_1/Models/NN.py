import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC

def to_categorical(Y):
    n_classes = len(np.unique(Y))
    Y_cat = np.zeros((Y.shape[0], n_classes))
    for i, y in enumerate(Y):
        Y_cat[i, int(y)] = 1
    return Y_cat

def shuffle(x, y, seed=77):
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(x))
    rng.shuffle(permutation)
    return x[permutation], y[permutation]

def split(X, Y, train_frac=0.8, seed=77):
    X, Y = shuffle(X, Y, seed=seed)
    upto = int(train_frac * X.shape[0])
    return (X[:upto, :], Y[:upto]), (X[upto:, :], Y[upto:])

def accuracy(Ytrue, preds):
    num_data_points = Ytrue.shape[0]
    correct_predictions = np.count_nonzero(Ytrue == preds)
    accuracy = correct_predictions / num_data_points
    return accuracy

def plot_loss_and_accuracy(history):
    xaxis = history['Epochs']
    plt.plot(xaxis, history['Train_loss'], label="Train loss")
    plt.plot(xaxis, history['Val_loss'], label="Validation loss")
    plt.plot(xaxis, history['Val_performance'], label="Validation accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.ylim(top=1, bottom=0)
    plt.show()

class Loss(ABC):
    @abstractmethod
    def __call__(self, true_labels, predictions):
        pass

    @abstractmethod
    def derivative(self, true_labels, predictions):
        pass

class SquaredLoss(Loss):
    def __call__(self, true_labels, predictions):
        diff = true_labels - predictions
        loss = diff.T @ diff
        return loss / 2

    def derivative(self, true_labels, predictions):
        return predictions - true_labels  # turn this around and do gradient ASCENT, works too ;)

class CrossEntropy(Loss):
    def __init__(self):
        self.softmax = Softmax()

    def __call__(self, true_labels, predictions):
        y = true_labels.argmax(axis=1) if true_labels.ndim > 1 else true_labels
        m = y.shape[0]
        p = self.softmax(predictions)
        log_likelihood = -np.log(p[list(range(m)), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def derivative(self, true_labels, predictions):
        y = true_labels.argmax(axis=1) if true_labels.ndim > 1 else true_labels
        m = y.shape[0]
        grad = self.softmax(predictions)
        grad[range(m), y] -= 1
        grad = grad / m
        return grad

class Activation(ABC):
    """
    Base class for all activation functions
    """

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

class Linear(Activation):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1

    def __str__(self):
        return "Linear"

class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self(x) * (1 - self(x))

    def __str__(self):
        return "Sigmoid"

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.maximum(self.alpha * x, x)

    def derivative(self, x):
        if not isinstance(x, np.ndarray):
            return 0 if x < 0 else 1
        x = x.copy()
        negative_dims = x < 0
        x[negative_dims] = self.alpha
        x[~negative_dims] = 1
        return x

    def __str__(self):
        return f"Leaky ReLU - alpha={self.alpha}"

class ReLU(LeakyReLU):
    def __init__(self):
        super().__init__(alpha=0)

    def __str__(self):
        return "ReLU"

class Sin(Activation):
    def __call__(self, x):
        return np.sin(x)

    def derivative(self, x):
        return np.cos(x)

    def __str__(self):
        return "Sine"

class Cos(Activation):
    def __call__(self, x):
        return np.cos(x)

    def derivative(self, x):
        return -np.sin(x)

    def __str__(self):
        return "Cosine"

class Softmax(Activation):
    def __call__(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def derivative(self, x):
        pass

    def __str__(self):
        return "Softmax"

class Tanh(Activation):
    def __call__(self, x):
        exp_x = np.exp(x)
        exp__x = np.exp(-x)
        return (exp_x - exp__x) / (exp_x + exp__x)

    def derivative(self, x):
        return 1 - self(x) ** 2

    def __str__(self):
        return "Tanh"

class MultiActivations(Activation):
    def __init__(self, dimensions, activations):
        self.activations = []
        neurons_per_act = int(np.ceil(dimensions / len(activations)))
        prev = 0
        for act in activations[:-1]:
            self.activations.append((act, prev, prev + neurons_per_act))
            prev += neurons_per_act
        self.activations.append((activations[-1], prev, dimensions))  # take the rest

    def __call__(self, z):
        x = z.copy()
        for act_func, s, e in self.activations:
            x[s:e] = act_func(x[s:e])
        return x

    def derivative(self, z):
        x = z.copy()
        for act_func, s, e in self.activations:
            x[s:e] = act_func.derivative(x[s:e])
        return x

    def __str__(self):
        return ", ".join([str(act) for act, _, _ in self.activations])

class Layer:
    def __init__(self, neuron_count: int, activation_func: Activation, next_layer=None, weight=None):
        self.n_neurons = neuron_count
        self.activation_func = activation_func
        self.pre_activation = None
        self.activation = None
        self.next_layer = next_layer
        self.gradient = None
        self.weight = weight  # weight matrix connecting to next layer

    def backward(self):
        d_activation = self.activation_func.derivative
        self.gradient = self.weight.W.T @ self.next_layer.gradient * d_activation(self.activation)

    def forward(self):
        raise NotImplementedError("The weight objects do all the forward pass job - Please call that method.")

    def __str__(self):
        return f"#Neurons: {self.n_neurons}"

class InputLayer(Layer):
    def __init__(self, neuron_count: int, *args, **kwargs):
        super().__init__(neuron_count, Linear())

class Weight:
    def __init__(self, prev_layer: Layer, next_layer: Layer):
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.dim_out, self.dim_in = next_layer.n_neurons, prev_layer.n_neurons
        self.W = np.random.randn(self.dim_out, self.dim_in) * 0.1
        self.bias = np.random.randn(self.dim_out, 1) * 0.1
        self.gradient_acc = np.zeros(self.W.shape)
        self.bias_acc = np.zeros(self.bias.shape)
        self.activation_func = self.next_layer.activation_func

    def forward(self, x):
        self.prev_layer.activation = x
        pre_activation = self.W @ x + self.bias
        self.next_layer.pre_activation = pre_activation
        self.next_layer.activation = self.activation_func(pre_activation)
        return self.next_layer.activation

    def backward(self, lr):
        self.bias_acc += lr * self.next_layer.gradient
        self.gradient_acc += lr * np.outer(self.next_layer.gradient, self.prev_layer.activation)

    def update(self, normalization):
        self.W = self.W - self.gradient_acc / normalization
        self.bias = self.bias - self.bias_acc / normalization
        self.gradient_acc = np.zeros(self.W.shape)
        self.bias_acc = np.zeros(self.bias.shape)

class NeuralNetwork:
    def __init__(self, layers: list):
        self.metric = None
        self.layers = layers
        self.weights = []
        self.y = None
        self.loss_f = None
        self.loss = -1
        self.lr = 0.01
        self.batch_size = None
        for i, layer in enumerate(self.layers[:-1]):
            weight = Weight(layer, self.layers[i + 1])
            self.weights.append(weight)
            layer.next_layer = self.layers[i + 1]
            layer.weight = weight

    def forward(self, x):
        activation = x
        for weight in self.weights:
            activation = weight.forward(activation)
        return activation

    def backprop(self):
        assert self.loss_f is not None and self.y is not None
        preds = self.layers[-1].activation
        loss = self.loss_f.derivative(self.y, preds)

        self.layers[-1].gradient = loss
        self.loss += float(self.loss_f(self.y, preds))
        rev_layers = list(reversed(self.layers))
        for layer in rev_layers[1:-1]:
            layer.backward()
        for weight in self.weights:
            weight.backward(self.lr)

    def compile(self, loss_function, metric):
        self.loss_f = loss_function
        self.metric = metric

    def update(self):
        for weight in self.weights:
            weight.update(self.batch_size)

    def fit(self, X, Y, X_val=None, Y_val=None, learning_rate=0.01, n_epochs=25, batch_size=64):
        self.lr = learning_rate
        self.batch_size = batch_size
        num_samples = X.shape[0]
        train_losses = []
        val_losses = []
        val_metrics = []
        for epoch in range(n_epochs):
            X, Y = shuffle(X, Y)
            self.loss = 0
            for i, (x, y) in enumerate(zip(X, Y)):
                self.y = y.reshape(-1, 1)
                self.forward(x.reshape(-1, 1))
                self.backprop()
                if i % batch_size == 0 and i != 0:
                    self.update()
            self.loss /= num_samples

            if X_val is not None and Y_val is not None:
                tr_loss, val_loss, val_metric = self.print_predict(X_val, Y_val, epoch=epoch)
                train_losses.append(tr_loss)
                val_losses.append(val_loss)
                val_metrics.append(val_metric)
            else:
                train_losses.append(self.loss)
                print("Epoch {} - Loss = {:.3f}".format(epoch, self.loss))
            return {"Train_loss": train_losses, "Val_loss": val_losses,
                "Val_performance": val_metrics, "Epochs": list(range(n_epochs))}

    def print_predict(self, X, Y, epoch=-1):
        if X is None or Y is None:
            return
        train_loss = self.loss
        preds = self.predict(X.T)
        val_loss = np.sum(self.loss_f(Y, preds)) / Y.shape[0]
        metric_val = self.metric(Y, preds)
        print("Epoch {} - Train loss = {:.3f} Validation Loss = {:.3f}, performance: {:.4f}"
              .format(epoch, train_loss, val_loss, metric_val))
        return train_loss, val_loss, metric_val

    def predict_proba(self, X):
        probs = self.forward(X)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=0)
        return preds
