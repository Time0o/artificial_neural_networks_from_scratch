import math
from itertools import product

import matplotlib
import numpy as np

from .dataset import plot_dataset
from .plotting import figsize, subplots
from .subsampling import split_samples
from .util import is_iterable, make_iterable


class TwoLayerPerceptron:
    def __init__(self, hidden_nodes, bias=True):
        self.hidden_nodes = hidden_nodes
        self.bias = bias

    def infer(self, inputs, threshold=True, train=False):
        assert hasattr(self, 'weights')

        if self.bias:
            inputs = self._bias(inputs)

        hidden_ = self.weights[0] @ inputs
        hidden = self._activate(hidden_)

        if self.bias:
            hidden = self._bias(hidden)

        outputs_ = self.weights[1] @ hidden
        outputs = self._activate(outputs_)

        if threshold:
            outputs = np.sign(outputs)

        if train:
            return inputs, hidden, outputs
        else:
            return outputs

    def train(self,
              inputs,
              labels,
              weights=None,
              momentum=None,
              learning_rate=0.001,
              momentum_alpha=0.9,
              epochs=20,
              batch=True):

        if weights is None:
            self.init_weights(inputs, labels)
        else:
            self.weights = [w.copy() for w in weights]

        if momentum is None:
            self.momentum = [0, 0]
        else:
            self.momentum = momentum

        for _ in range(epochs):
            if batch:
                self._train(inputs, labels, learning_rate, momentum_alpha)
            else:
                for sample, label in zip(inputs.T, labels.T):
                    self._train(sample.reshape(inputs.shape[0], 1),
                                label.reshape(labels.shape[0], 1),
                                learning_rate,
                                momentum_alpha)

        self._normalize_weights()

    def error(self, inputs, labels):
        predictions = self.infer(inputs, threshold=False)

        return np.sum((predictions - labels)**2) / inputs.shape[1]

    def precision(self, inputs, labels):
        predictions = self.infer(inputs)

        correct = 0
        for pred, label in zip(predictions.T, labels.T):
            correct += np.all(pred == label)

        return correct / inputs.shape[1]

    def init_weights(self, inputs, labels):
        self.weights = [
            np.random.normal(size=(self.hidden_nodes,
                                   inputs.shape[0] + (1 if self.bias else 0))),

            np.random.normal(size=(labels.shape[0],
                                   self.hidden_nodes + (1 if self.bias else 0)))
        ]

    def _train(self, inputs, labels, learning_rate, momentum_alpha):
        inputs_, hidden, outputs = self.infer(
            inputs, threshold=False, train=True)

        # backward pass
        d_outputs = self._activate_derive(outputs)
        delta_outputs = (outputs - labels) * d_outputs

        d_hidden = self._activate_derive(hidden)
        delta_hidden = self.weights[1].T @ delta_outputs * d_hidden
        delta_hidden = delta_hidden[:self.hidden_nodes, :]

        # weight update
        self.momentum[0] = momentum_alpha * self.momentum[0] - \
                           (1 - momentum_alpha) * delta_hidden @ inputs_.T

        self.momentum[1] = momentum_alpha * self.momentum[1] - \
                           (1 - momentum_alpha) * delta_outputs @ hidden.T

        self.weights[0] += learning_rate * self.momentum[0]
        self.weights[1] += learning_rate * self.momentum[1]

    def _normalize_weights(self):
        pass

        #for weight in self.weights:
        #    for row in range(weight.shape[0]):
        #        weight[row, :] /= la.norm(weight[row, :])

        #for weight in self.weights:
        #    for col in range(weight.shape[1]):
        #        weight[:, col] /= la.norm(weight[:, col])

        #for weight in self.weights:
        #    weight /= np.sum(weight)

    @staticmethod
    def _bias(values):
        return np.vstack((values, np.ones(values.shape[1])))

    @staticmethod
    def _activate(values):
        return 2 / (1 + np.exp(-values)) - 1

    @staticmethod
    def _activate_derive(activation):
        return ((1 + activation) * (1 - activation)) / 2


def find_converging_model(inputs,
                          labels,
                          hidden_nodes,
                          learning_rate,
                          momentum_alpha,
                          epochs,
                          runs):

    model = TwoLayerPerceptron(hidden_nodes=hidden_nodes)

    successes = 0

    least_epochs = math.inf
    best_model = None

    for _ in range(runs):
        weights = None
        momentum = None
        for epoch in range(epochs):
            model.train(inputs,
                        labels,
                        weights=weights,
                        momentum=momentum,
                        learning_rate=learning_rate,
                        momentum_alpha=momentum_alpha,
                        epochs=1)

            weights = model.weights
            momentum = model.momentum

            if model.precision(inputs, labels) == 1:
                successes += 1

                if epoch < least_epochs:
                    least_epochs = epoch
                    best_model = model

                break

    print("Convergence achieved on {} of {} runs".format(successes, runs))

    return best_model


def plot_hidden_node_influence(inputs,
                               labels,
                               hidden_nodes,
                               learning_rate,
                               momentum_alpha,
                               epochs,
                               runs):

    #_, (ax1, ax2) = subplots(1, 2)
    _, ax1 = subplots(1, 1)
    ax2 = ax1.twinx()

    errors = []
    precisions = []

    for hn in make_iterable(hidden_nodes):
        errors_ = []
        precisions_ = []

        for _ in range(runs):
            model = TwoLayerPerceptron(hidden_nodes=hn)

            model.train(inputs,
                        labels,
                        learning_rate=learning_rate,
                        momentum_alpha=momentum_alpha,
                        epochs=epochs)

            errors_.append(model.error(inputs, labels))
            precisions_.append(model.precision(inputs, labels))

        errors.append(errors_)
        precisions.append(precisions_)

    ax1.errorbar(hidden_nodes,
                 [np.mean(e) for e in errors],
                 yerr=[np.std(e) for e in errors],
                 color='r')

    ax2.errorbar(hidden_nodes,
                 [np.mean(p) for p in precisions],
                 yerr=[np.std(p) for p in precisions],
                 color='g')

    ax1.set_xlabel("Hidden Nodes")
    ax1.set_ylabel("MSE", color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.grid()

    ax2.set_ylabel("Accuracy", color='g')
    ax2.tick_params(axis='y', labelcolor='g')


def plot_validation_error(inputs,
                          labels,
                          subsamplings,
                          hidden_nodes,
                          learning_rate,
                          momentum_alpha,
                          epochs,
                          runs,
                          batch):

    if not is_iterable(hidden_nodes):
        hidden_nodes = [hidden_nodes]

    _, axes = subplots(len(subsamplings), len(hidden_nodes))

    axes = axes.reshape(len(subsamplings), len(hidden_nodes))

    color1, color2 = matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][:2]

    for i, subsampling in enumerate(subsamplings):
        (inputs_train, labels_train), (inputs_val, labels_val) = \
            split_samples(inputs,
                          labels,
                          subsampling.selectors,
                          subsampling.fractions)

        for j, hn in enumerate(make_iterable(hidden_nodes)):
            for _ in range(runs):
                model = TwoLayerPerceptron(hidden_nodes=hn)
                model.init_weights(inputs, labels)

                weights = model.weights
                momentum = None

                error_train = [model.error(inputs_train, labels_train)]
                error_val = [model.error(inputs_val, labels_val)]

                for _ in range(epochs):
                    model.train(inputs_train,
                                labels_train,
                                weights=weights,
                                momentum=momentum,
                                learning_rate=learning_rate,
                                momentum_alpha=momentum_alpha,
                                epochs=1,
                                batch=batch)

                    error_train.append(model.error(inputs_train, labels_train))
                    error_val.append(model.error(inputs_val, labels_val))

                    weights = model.weights
                    momentum = model.momentum

                axes[i, j].plot(range(epochs + 1), error_train, color=color1)
                axes[i, j].plot(range(epochs + 1), error_val, color=color2)

            axes[i, j].plot([], color=color1, label="Training Set")
            axes[i, j].plot([], color=color2, label="Validation Set")

            axes[i, j].set_xlabel("Epoch")
            axes[i, j].set_ylabel("MSE")

            axes[i, j].legend()
            axes[i, j].grid()


def plot_learning_curve(inputs,
                        labels,
                        hidden_nodes,
                        learning_rate,
                        momentum_alpha,
                        epochs):

    _, ax = subplots(1, 1)

    if not is_iterable(hidden_nodes):
        hidden_nodes = [hidden_nodes]

    if not is_iterable(learning_rate):
        learning_rate = [learning_rate]

    for hn, lr in product(hidden_nodes, learning_rate):
        model = TwoLayerPerceptron(hidden_nodes=hn)
        weights = None
        momentum = None

        errors = []
        for _ in range(epochs):
            model.train(inputs,
                        labels,
                        weights=weights,
                        momentum=momentum,
                        learning_rate=lr,
                        momentum_alpha=momentum_alpha,
                        epochs=1)

            weights = model.weights
            momentum = model.momentum

            errors.append(model.error(inputs, labels))

        fmt = r"{:d} hidden nodes, $\eta = {:1.2e}$"
        ax.semilogy(errors, label=fmt.format(hn, lr))

    ax.set_title("Learning Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")

    ax.legend()
    ax.grid()


def plot_generalization_performance(inputs,
                                    labels,
                                    split,
                                    hidden_nodes,
                                    learning_rate,
                                    momentum_alpha,
                                    epochs,
                                    runs):

    _, ax = subplots(1, 1)

    n = inputs.shape[1]
    n_train = int(split * n)

    errors_train = []
    errors_val = []

    for hn in hidden_nodes:
        errors_train_ = []
        errors_val_ = []

        for _ in range(runs):
            shuffle = np.random.permutation(n)

            inputs_train = inputs[:, shuffle][:, :n_train].copy()
            inputs_val = inputs[:, shuffle][:, n_train:].copy()

            labels_train = labels[:, shuffle][:, :n_train].copy()
            labels_val = labels[:, shuffle][:, n_train:].copy()

            model = TwoLayerPerceptron(hidden_nodes=hn)

            model.train(inputs_train,
                        labels_train,
                        learning_rate=learning_rate,
                        epochs=epochs)

            errors_train_.append(model.error(inputs_train, labels_train)),
            errors_val_.append(model.error(inputs_val, labels_val))

        errors_train.append(errors_train_)
        errors_val.append(errors_val_)

    for errors, label in ((errors_train, "Training Set"),
                          (errors_val, "Validation Set")):

        median = [np.median(e) for e in errors]
        ax.semilogy(hidden_nodes, median, label=label)

    ax.set_xlabel("Number of Hidden Nodes")
    ax.set_ylabel("MSE")

    ax.grid()
    ax.legend()


def plot_decision_boundary(model,
			   inputs,
			   labels,
			   resolution=1000):

	n = resolution**2

	_, ax = subplots(1, 1)

	plot_dataset(inputs, labels, bias=False, ax=ax)

	x = np.linspace(*ax.get_xlim(), resolution)
	y = np.linspace(*ax.get_ylim(), resolution)

	xx, yy = np.meshgrid(x, y)

	inputs = np.vstack((xx.reshape(1, n), yy.reshape(1, n)))
	labels = model.infer(inputs).reshape(resolution, resolution)

	ax.contour(xx, yy, labels, colors='r')
