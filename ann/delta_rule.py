import sys

import numpy as np

from .plotting import subplots
from .util import make_iterable


def delta_evaluate(sample, weights):
    return np.dot(weights, sample)


def delta_classify(sample, weights):
    s = np.sign(delta_evaluate(sample, weights))

    if s >= 0:
        return 1
    else:
        return -1


def delta_error(inputs, labels, weights):
    err = 0
    for sample, label in zip(inputs.T, labels.T):
        err += (delta_evaluate(sample, weights) - label)**2

    return err / 2


def delta_precision(inputs, labels, weights):
    correct = 0
    for sample, label in zip(inputs.T, labels.T):
        correct += (delta_classify(sample, weights) == label)

    return correct / inputs.shape[1]


def delta_init_weights(inputs, labels):
    size = (labels.shape[0], inputs.shape[0])
    radius = 1 / np.sqrt(inputs.shape[1])

    return radius * np.random.normal(size=size)


def delta_train(inputs,
                labels,
                weights=None,
                learning_rate=0.001,
                epochs=20,
                batch=True,
                benchmark=False):

    if weights is None:
        weights = delta_init_weights(inputs, labels)

    if batch:
        for _ in range(epochs):
            weights -= learning_rate * (weights @ inputs - labels) @ inputs.T
    else:
        for _ in range(epochs):
            for sample, label in zip(inputs.T, labels.T):
                weights -= learning_rate * (np.dot(weights, sample) - label) * sample

    if benchmark:
        error = delta_error(inputs, labels, weights)
        precision = delta_precision(inputs, labels, weights)

        return weights, error, precision
    else:
        return weights


def delta_plot_convergence(inputs,
                           labels,
                           learning_rates,
                           epochs,
                           batch,
                           axes=None):
    if axes is None:
        _, (ax1, ax2) = subplots(2, 1)
    else:
        ax1, ax2 = axes

    weights_init = delta_init_weights(inputs, labels)
    for lr in learning_rates:
        errors = []
        precisions = []

        weights = weights_init.copy()
        for ep in epochs:
            weights, error, precision = delta_train(
                inputs, labels, weights,
                learning_rate=lr, epochs=ep,
                batch=batch, benchmark=True)

            errors.append(error)
            precisions.append(precision)

        label = r"$\eta = {:1.1e}$".format(lr)
        ax1.plot(epochs, errors, label=label)
        ax2.plot(epochs, precisions, label=label)

    ax1.axhline(0, linestyle='--', color='r')
    ax2.axhline(1, linestyle='--', color='r')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SOS Error")
    ax1.legend()
    ax1.grid()

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid()


def delta_compare_convergence_speed(inputs, labels, learning_rates, runs):
    LEARNING_RATE_BATCH_MAX = 0.001

    for lr in make_iterable(learning_rates):
        epochs_batch = []
        epochs_seq = []

        for _ in range(runs):
            weights_batch = delta_init_weights(inputs, labels)
            weights_seq = weights_batch.copy()

            if lr <= LEARNING_RATE_BATCH_MAX:
                ep = 1
                while True:
                    weights_batch, error, precision = delta_train(
                        inputs, labels, weights_batch,
                        learning_rate=lr, epochs=1,
                        batch=True, benchmark=True)

                    if precision == 1:
                        epochs_batch.append(ep + 1)
                        break

                    ep += 1

            ep = 1
            while True:
                weights_seq, error, precision = delta_train(
                    inputs, labels, weights_seq,
                    learning_rate=lr, epochs=1,
                    batch=False, benchmark=True)

                if precision == 1:
                    epochs_seq.append(ep + 1)
                    break

                ep += 1

        if lr <= LEARNING_RATE_BATCH_MAX:
            fmt = "eta = {}: batch learning converged after {:.2} +/- {:.2} epochs"
            print(fmt.format(lr, np.mean(epochs_batch), np.std(epochs_batch)))

        fmt = "eta = {}: sequential learning converged after {:.2} +/- {:.2} epochs"
        print(fmt.format(lr, np.mean(epochs_seq), np.std(epochs_seq)))

        sys.stdout.flush()


def delta_plot_weight_sensitivity(inputs,
                                  labels,
                                  learning_rates,
                                  eps_convergence,
                                  runs,
                                  batch,
                                  axes=None):

    if axes is None:
        _, axes = subplots(2, 2)
        axes = axes.flatten()

    for lr, ax in zip(learning_rates, axes):
        epochs_till_convergence = []

        for _ in range(runs):
            weights = delta_init_weights(inputs, labels)

            epoch = 0
            error_last = None
            while True:
                weights, error, precision = delta_train(
                    inputs, labels, weights,
                    learning_rate=lr, epochs=1,
                    batch=batch, benchmark=True)

                if error_last is not None:
                    if abs(error - error_last) < eps_convergence:
                        break

                error_last = error

                epoch += 1

            epochs_till_convergence.append(epoch)

        ax.hist(epochs_till_convergence)

        ax.set_title(r"$\eta = {}$".format(lr))
        ax.set_xlabel("Epochs Until Convergence")
