import numpy as np

from .dataset import LABEL_A, LABEL_B


class Subsampling:
    def __init__(self, name, selectors, fractions):
        if not isinstance(selectors, list):
            selectors = [selectors]

        if not isinstance(fractions, list):
            fractions = [fractions]

        self.name = name
        self.selectors = selectors
        self.fractions = fractions


SUBSAMPLINGS = [
    Subsampling("Nothing",
                lambda sample, label: False, 0.0),
    Subsampling("25% from each Dataset",
                lambda sample, label: True, 0.25),
    Subsampling("50% from Dataset A",
                lambda sample, label: label == LABEL_A, 0.5),
    Subsampling("50% from Dataset B",
                lambda sample, label: label == LABEL_B, 0.5),
    Subsampling("20%/80% from Dataset A",
                [lambda sample, label: label == LABEL_A and sample[0] < 0,
                 lambda sample, label: label == LABEL_A and sample[0] > 0],
                [0.2, 0.8])
]


def split_samples(inputs, labels, selectors, fractions):
    if not isinstance(selectors, list):
        selectors = [selectors]

    if not isinstance(fractions, list):
        fractions = [fractions]

    selected = [[] for _ in range(len(selectors))]
    for i, (sample, label) in enumerate(zip(inputs.T, labels.T)):
        for j, selector in enumerate(selectors):
            if selector(sample, label):
                selected[j].append(i)

    for subset in selected:
        np.random.shuffle(subset)

    remove = []
    for subset, fraction in zip(selected, fractions):
        remove += subset[:int(fraction * len(subset))]

    mask = np.ones(inputs.shape[1], dtype=bool)
    mask[remove] = False

    return ((inputs[:, mask], labels[:, mask]),
            (inputs[:, ~mask], labels[:, ~mask]))
