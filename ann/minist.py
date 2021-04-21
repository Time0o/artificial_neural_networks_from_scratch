import numpy as np

from .plotting import figsize, subplots


def load_data(fname):
    with open(fname, 'r') as f:
        data = np.genfromtxt(f, delimiter=',', dtype=np.uint8)

    return data.astype(np.float32)


def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)


def preview_data(digits, targets, unique=True, n=(3, 3), scale=.5, axes=None):
    if unique:
        n = (3, 3)

        digits_ = digits
        digits = []

        for d in range(10):
            i = np.random.choice(np.where(targets == d)[0])
            digits.append(digits_[i])

    if axes is None:
        _, axes = subplots(n[0], n[1], size=figsize(scale * n[0], scale * n[1]))

    for d, t, ax in zip(digits, targets, axes.flatten()):
        ax.imshow(d.reshape(28, 28), cmap='gray')
        ax.set_title(t)
        ax.axis('off')


def split_data(digits, targets, frac):
    # shuffle data
    i = np.random.permutation(digits.shape[0])

    digits_shuffled = digits[i, :]
    targets_shuffled = targets[i]

    # split data
    n1 = int(frac * digits.shape[0])

    d1 = digits_shuffled[:n1, :]
    d2 = digits_shuffled[n1:, :]

    t1 = targets_shuffled[:n1]
    t2 = targets_shuffled[n1:]

    return d1, t1, d2, t2


def digit_occurrences(targets, ax=None):
    if ax is None:
        _, ax = subplots(1, 1)

    ax.hist(targets, bins=np.arange(11) - 0.5, rwidth=0.7)
    ax.set_xticks(range(10))

    ax.set_xlabel("Digit")
    ax.set_ylabel("Occurrences")


def train_test_split(inputs, labels, frac=0.8):
    n = inputs.shape[0]
    n_train = int(frac * n)

    i = np.random.permutation(n)

    return inputs[i, :][:n_train, :], labels[i][:n_train], \
           inputs[i, :][n_train:, :], labels[i][n_train:]


def balanced_subsampling(inputs, labels, frac=1.0):
    counts = {
        l: int(round(frac * c))
        for l, c in zip(*np.unique(labels, return_counts=True))
    }

    inputs_subs = []
    labels_subs = []

    for l, c in counts.items():
        i = np.random.choice(np.where(labels == l)[0], size=c, replace=False)

        inputs_subs.append(inputs[i, :])
        labels_subs.append(labels[i])

    inputs_subs = np.concatenate(inputs_subs)
    labels_subs = np.concatenate(labels_subs)

    i = np.random.permutation(inputs_subs.shape[0])

    return inputs_subs[i, :], labels_subs[i]
