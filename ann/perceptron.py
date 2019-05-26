import numpy as np

from .dataset import plot_dataset, plot_separator
from .plotting import subplots
from .subsampling import split_samples


def perceptron_classify(sample, weights):
    s = np.sign(np.dot(weights, sample))

    if s >= 0:
        return 1
    else:
        return -1


def perceptron_precision(inputs, labels, weights):
    correct = 0
    for sample, label in zip(inputs.T, labels.T):
        correct += (perceptron_classify(sample, weights) == label)

    return correct / inputs.shape[1]


def perceptron_train(inputs,
                     labels,
                     init_weights=True,
                     learning_rate=0.01,
                     iterations=None,
                     stop_early=True,
                     benchmark=False):

    assert len(labels.shape) == 1 or labels.shape[0] == 1

    weights = np.zeros((1, inputs.shape[0]))

    if init_weights:
        weights = (np.sign(labels) @ inputs.T) / inputs.shape[1]

    if iterations == 0:
        if benchmark:
            return weights, perceptron_precision(inputs, labels, weights)
        else:
            return weights

    i = 0
    done = False
    while True:
        for j in range(inputs.shape[1]):
            # update weights
            dp = np.dot(weights, inputs[:, j])

            sign = np.sign(labels[:, j])
            if sign == -1 and dp >= 0:
                weights -= learning_rate * inputs[:, j].T
            elif sign == 1 and dp < 0:
                weights += learning_rate * inputs[:, j].T

            # determine number of currently correctly classified samples
            if stop_early:
                correct = perceptron_precision(inputs, labels, weights)

            # optionally stop if training has converged
            if stop_early and correct == 1.0:
                done = True
                break

            i += 1
            if i == iterations:
                done = True
                break

        if done:
            break

    if benchmark:
        return weights, perceptron_precision(inputs, labels, weights)
    else:
        return weights


def perceptron_plot_convergence(dataset_creator,
                                iterations,
                                runs,
                                ax=None):
    if ax is None:
        _, ax = subplots(1, 1)

    for _ in range(runs):
        inputs, labels = dataset_creator()

        precisions = []
        for it in iterations:
            _, precision = perceptron_train(inputs,
                                            labels,
                                            init_weights=False,
                                            iterations=it,
                                            benchmark=True)
            precisions.append(precision)

        ax.plot(iterations, precisions)

    ax.axhline(1, linestyle='--', color='r')

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")

    ax.grid()


def perceptron_plot_generalization(inputs,
                                   labels,
                                   subsamplings,
                                   iterations,
                                   runs):

    _, axes = subplots(len(subsamplings), 3)

    for i, subsampling in enumerate(subsamplings):
        precs_a = []
        precs_b = []

        weights = []
        for j in range(runs):
            (inputs_sub, labels_sub), (_, _) = split_samples(
                inputs, labels, subsampling.selectors, subsampling.fractions)

            weights_ = perceptron_train(inputs_sub,
                                        labels_sub,
                                        iterations=iterations)

            label_a, label_b = np.unique(labels)
            idx_a = (labels == label_a)[0]
            idx_b = (labels == label_b)[0]

            prec_a = perceptron_precision(
                inputs[:, idx_a], labels[:, idx_a], weights_)

            prec_b = perceptron_precision(
                inputs[:, idx_b], labels[:, idx_b], weights_)

            weights.append(weights_)
            precs_a.append(prec_a)
            precs_b.append(prec_b)

        plot_dataset(inputs_sub, labels_sub, ax=axes[i, 0])

        plot_separator(inputs, labels, weights, ax=axes[i, 1])

        axes[i, 2].plot([0, 1], [0.5, 0.5], color='k')
        axes[i, 2].plot([0.5, 0.5], [0, 1], color='k')
        axes[i, 2].scatter(precs_a, precs_b)

        axes[i, 2].set_xlabel("Class A")
        axes[i, 2].set_ylabel("Class B")

        axes[i, 2].grid()
