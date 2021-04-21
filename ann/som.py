import numpy as np
import numpy.linalg as la


def som(inputs,
        outputs,
        learning_rate=0.2,
        epochs=20,
        neighbourhood='simple',
        neighbourhood_init=50):

    if neighbourhood not in ['simple', 'cyclic', 'grid']:
        raise ValueError("invalid neighbourhood type")

    n_float = neighbourhood_init
    n_step = neighbourhood_init / epochs

    if neighbourhood == 'grid':
        output_x, output_y = np.mgrid[:outputs[1], :outputs[0]]

        weights = np.random.uniform(
            0, 1, size=(np.prod(outputs), inputs.shape[1]))
    else:
        output_ind = list(range(outputs))

        weights = np.random.uniform(
            0, 1, size=(outputs, inputs.shape[1]))

    for _ in range(epochs):
        for row in inputs:
            i = np.argmin(la.norm(weights - row, axis=1))

            n_r = int(n_float)
            n_low = i - n_r
            n_high = i + n_r + 1

            if neighbourhood == 'grid':
                i_x, i_y = np.unravel_index(i, outputs)

                n = np.abs(output_x - i_x) + np.abs(output_y - i_y) <= n_r
                ind = np.ravel_multi_index(np.where(n), outputs)

            elif neighbourhood == 'cyclic':
                if n_low < 0:
                    ind = output_ind[n_low:] + output_ind[:n_high]
                elif n_high > outputs:
                    ind = output_ind[n_low:] + output_ind[:(n_high - outputs)]
                else:
                    ind = output_ind[n_low:n_high]

            else:
                n_low = max(0, n_low)
                n_high = min(outputs, n_high)

                ind = slice(n_low, n_high)

            delta = learning_rate * (row - weights[ind, :])
            weights[ind, :] += delta

        n_float -= n_step

    scores = [np.argmin(la.norm(weights - row, axis=1)) for row in inputs]

    return np.array(scores), np.argsort(scores), weights
