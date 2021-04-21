import numpy as np


def binary_patterns(length):
    for x in range(1 << length):
        bin_str = bin(x)[2:].zfill(8)

        yield np.array([1 if b == '1' else -1 for b in bin_str], dtype=np.int8)


def sparse_patterns(size, n, activity=0.5):
    ones = int(activity * size * n)

    patterns = np.zeros(n * size, dtype=np.int8)

    i = np.random.choice(len(patterns), size=ones, replace=False)
    patterns[i] = 1

    return np.hsplit(patterns, n)
