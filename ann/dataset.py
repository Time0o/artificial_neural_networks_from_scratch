import numpy as np

from .plotting import subplots


LABEL_A = -1
LABEL_B = 1


def randn2D(mu, sigma, n):
    sigma = [[sigma**2, 0], [0, sigma**2]]

    return np.random.multivariate_normal(mu, sigma, size=n)


def randn2D_mirrored(mu, sigma, n):
    x1 = sigma * np.random.randn(1, n // 2) - mu[0]
    x2 = sigma * np.random.randn(1, n // 2) + mu[0]

    x = np.vstack((x1.T, x2.T))
    y = sigma * np.random.randn(n, 1) + mu[1]

    return np.hstack((x, y))


def create_dataset(n, params, labels, bias=True, mirror=None):
    datasets = []
    for i, (mu, sigma) in enumerate(params):
        if mirror is not None and mirror[i]:
            datasets.append(randn2D_mirrored(mu, sigma, n))
        else:
            datasets.append(randn2D(mu, sigma, n))

    datasets_labelled = []
    for ds, l in zip(datasets, labels):
        datasets_labelled.append(np.hstack((ds, np.full((ds.shape[0], 1), l))))

    dataset = np.concatenate(datasets_labelled)

    np.random.shuffle(dataset)

    patterns = dataset[:, :2].T

    if bias:
        patterns = np.vstack((patterns, np.ones((1, dataset.shape[0]))))

    targets = dataset[:, -1, np.newaxis].T

    return patterns, targets


def plot_dataset(inputs, labels, bias=True, ax=None):
    if ax is None:
        _, ax = subplots(1, 1)

    if bias:
        inputs = inputs[:-1, :]

    for i, label in enumerate(np.unique(labels)):
        x, y = np.vsplit(inputs[:, np.where(labels == label)[1]], 2)
        ax.scatter(x, y, label="Class {}".format(chr(ord('A') + i)))

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.legend()


def plot_separator(inputs, labels, weights, bias=True, ax=None):
    if ax is None:
        _, ax = subplots(1, 1)

    plot_dataset(inputs, labels, bias=bias, ax=ax)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmin, xmax = xlim
    ymin, ymax = ylim

    for w in weights:
        if bias:
            w1, w2, w0 = w[0, :]
        else:
            w1, w2 = w[0, :]
            w0 = 0

        y0 = -(w1 * xmin + w0) / w2
        y1 = -(w1 * xmax + w0) / w2

        if y0 < ymin:
            y0 = ymin
            x0 = -(w0 + w2 * ymin) / w1
        elif y0 > ymax:
            y0 = ymax
            x0 = -(w0 + w2 * ymax) / w1
        else:
            x0 = xmin

        if y1 < ymin:
            y1 = ymin
            x1 = -(w0 + w2 * ymin) / w1
        elif y1 > ymax:
            y1 = ymax
            x1 = -(w0 + w2 * ymax) / w1
        else:
            x1 = xmax

        ax.plot([x0, x1], [y0, y1],
                color='r', alpha=(0.5 if len(weights) > 1 else 1))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
