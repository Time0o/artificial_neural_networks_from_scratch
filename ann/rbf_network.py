import math
import os; os.environ['KERAS_BACKEND'] = 'theano'
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from scipy.stats import multivariate_normal

from .plotting import figsize, subplots
from .util import make_iterable, pop_val

DEFAULT_NODES = 10
DEFAULT_NODE_POS = 'equal'
DEFAULT_NODE_STD = 1
DEFAULT_NODE_PAD = 0
DEFAULT_CL_SYMMETRICAL = True
DEFAULT_CL_LEARN_COV = True
DEFAULT_CL_RANDOM_START = True
DEFAULT_CL_LEARNING_RATE = 0.2
DEFAULT_CL_EPOCHS = 100
DEFAULT_CL_BIAS = False
DEFAULT_CL_BIAS_WEIGHT = 0.01
DEFAULT_CL_VERBOSE = False
DEFAULT_LEARNING_RATE_DELTA = 0.1
DEFAULT_EPOCHS_DELTA = 100
DEFAULT_SHUFFLE_DELTA = True
DEFAULT_POSTPROCESSING = None
DEFAULT_TRAINING = 'least_squares'
DEFAULT_RUNS = 10

ALPHA = 0.5
LINEWIDTH = 3


class GaussianRBFNetwork:
    SING_VAR_DIV = 10
    SING_VAR_MUL = 1.1

    def __init__(self,
                 domain,
                 nodes=DEFAULT_NODES,
                 node_pos=DEFAULT_NODE_POS,
                 node_std=DEFAULT_NODE_STD,
                 node_pad=DEFAULT_NODE_PAD,
                 cl_symmetrical=DEFAULT_CL_SYMMETRICAL,
                 cl_learn_cov=DEFAULT_CL_LEARN_COV,
                 cl_random_start=DEFAULT_CL_RANDOM_START,
                 cl_learning_rate=DEFAULT_CL_LEARNING_RATE,
                 cl_epochs=DEFAULT_CL_EPOCHS,
                 cl_bias=DEFAULT_CL_BIAS,
                 cl_bias_weight=DEFAULT_CL_BIAS_WEIGHT,
                 cl_verbose=DEFAULT_CL_VERBOSE,
                 postprocessing=DEFAULT_POSTPROCESSING):

        if nodes < 2:
            raise ValueError("number of nodes must be > 1")

        self.domain = domain
        self.nodes = nodes
        self.node_pos = node_pos
        self.postprocessing = postprocessing

        if self.node_pos == 'cl':
            self.init_kernels_cl(std=node_std,
                                 symmetrical=cl_symmetrical,
                                 learn_cov=cl_learn_cov,
                                 random_start=cl_random_start,
                                 learning_rate=cl_learning_rate,
                                 epochs=cl_epochs,
                                 bias=cl_bias,
                                 bias_weight=cl_bias_weight,
                                 verbose=cl_verbose)

        elif self.node_pos == 'random':
            self.init_kernels_random(node_std)

        elif self.node_pos == 'equal':
            self.init_kernels_equal(node_std)

        elif self.node_pos == 'padded':
            self.init_kernels_padded(node_std, node_pad)

        elif self.node_pos == 'manual':
            pass

        else:
            raise ValueError("invalid node position")

        self.init_weights()

    def init_weights(self):
        self.weights = np.random.randn(self.nodes, 1)

    def init_kernels_equal(self, std=DEFAULT_NODE_STD):
        assert len(np.squeeze(self.domain).shape) == 1

        low, high = self.domain[0], self.domain[-1]
        self.mu = np.linspace(low, high, self.nodes)

        self.cov = np.full(self.nodes, std * std)

    def init_kernels_padded(self, std=DEFAULT_NODE_STD, pad=DEFAULT_NODE_PAD):
        assert len(np.squeeze(self.domain).shape) == 1

        low, high = self.domain[0] + pad, self.domain[-1] - pad
        self.mu = np.linspace(low, high, self.nodes)

        self.cov = np.full(self.nodes, std * std)

    def init_kernels_random(self, std=DEFAULT_NODE_STD):
        self.mu = np.array([self._randsample() for _ in range(self.nodes)])

        self.cov = np.full(self.nodes, std * std)

    def init_kernels_cl(self,
                        std=DEFAULT_NODE_STD,
                        symmetrical=DEFAULT_CL_SYMMETRICAL,
                        learn_cov=DEFAULT_CL_LEARN_COV,
                        random_start=DEFAULT_CL_RANDOM_START,
                        learning_rate=DEFAULT_CL_LEARNING_RATE,
                        epochs=DEFAULT_CL_EPOCHS,
                        bias=DEFAULT_CL_BIAS,
                        bias_weight=DEFAULT_CL_BIAS_WEIGHT,
                        verbose=DEFAULT_CL_VERBOSE):

        if random_start:
            self.init_kernels_random(std)

        winnings = np.zeros((self.mu.shape[0], 1), dtype=int)

        if bias:
            biases = np.full((self.mu.shape[0], 1), 1 / self.nodes)

        samples_ = self.domain.copy()

        for i in range(epochs):
            np.random.shuffle(samples_)

            for sample in samples_:
                err = sample - self.mu

                d = np.sum(err * err, axis=1).reshape(self.nodes, 1)
                if bias:
                    d -= bias_weight * biases

                i = np.argmin(d)

                delta = learning_rate * err[i, :]
                self.mu[i, :] += delta

                winnings[i] += 1

                if bias:
                    biases = 1 / self.nodes - winnings / (i + 1)

        if verbose:
            print("CL summary:\n")

            print("Positions are:")
            for pos in self.mu:
                print("x: {:.3f}, y: {:.3f}".format(*pos))

            print("\nWinnings are:")
            for pos, wins in zip(self.mu, winnings):
                print("x: {:.3f}, y: {:.3f} => {}".format(*pos, wins[0]))

        # adjust kernel covariances
        if learn_cov:
            if len(np.squeeze(self.domain).shape) == 1:
                err = "covariance learning not supported for 1D inputs"
                raise ValueError(err)

            cov = np.array([np.eye(self.domain.shape[1])
                            for _ in range(self.nodes)])

            radius = std * np.sqrt(-2 * np.log(1 - 0.9))

            for i, mu in enumerate(self.mu):
                distances = np.sqrt(np.sum((self.domain - mu)**2, axis=1))
                points = self.domain[distances < radius, :]

                if len(points) <= 1:
                    cov_ = std * std / self.SING_VAR_DIV * np.eye(2)
                else:
                    cov_ = np.cov(points.T)

                    if symmetrical:
                        cov_ = max(cov_[0, 0], cov_[1, 1]) * np.eye(2)
                    else:
                        if la.cond(cov_) >= 1 / sys.float_info.epsilon:
                            cov_[0, 0] *= self.SING_VAR_MUL
                            cov_[1, 1] *= self.SING_VAR_MUL

                cov[i, :, :] = cov_

            self.cov = cov

    def least_squares_fit(self, inputs, outputs):
        phi = self._gauss_matrix(inputs)

        self.weights, _, _, _ = la.lstsq(phi, np.ravel(outputs), rcond=None)
        self.weights = self.weights[:, np.newaxis]

    def delta_rule_train(self,
                         inputs,
                         outputs,
                         learning_rate=DEFAULT_LEARNING_RATE_DELTA,
                         epochs=DEFAULT_EPOCHS_DELTA,
                         shuffle=DEFAULT_SHUFFLE_DELTA,
                         improve=False):

        if not improve:
            self.init_weights()

        for _ in range(epochs):
            if shuffle:
                reorder = np.random.permutation(inputs.shape[0])
                inputs_ = inputs[reorder, :]
                outputs_ = outputs[reorder, :]
            else:
                inputs_ = inputs
                outputs_ = outputs

            for sample, expected in zip(inputs_, outputs_):
                phi = self._gauss_vector(sample)

                err = expected - phi.T @ self.weights

                self.weights += learning_rate * err * phi

    def predict(self, inputs):
        if self.weights is None:
            raise ValueError("weights not initialized")

        pred = self._gauss_matrix(inputs) @ self.weights

        if self.postprocessing is not None:
            pred = self.postprocessing(pred)

        return pred

    def residual(self, inputs, expected):
        pred = self.predict(inputs)

        return np.mean(np.abs(pred - expected))

    def error(self, inputs, expected):
        pred = self.predict(inputs)

        return np.mean((pred - expected)**2)

    def _randsample(self):
        return self.domain[np.random.randint(self.domain.shape[0]), :]

    def _gauss(self, x, i):
        return multivariate_normal.pdf(x, mean=self.mu[i], cov=self.cov[i])

    def _gauss_vector(self, sample):
        return np.array([self._gauss(sample, i)
                         for i in range(self.nodes)])[:, np.newaxis]

    def _gauss_matrix(self, inputs):
        return np.array([
            [self._gauss(x, i) for i in range(self.nodes)] for x in inputs
        ])


class GaussianRBFDataset:
    def __init__(self, x, y, x_val, y_val, x_test, y_test):
        self.x = x
        self.y = y
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test


def create_test_dataset(noise_variance=None, show=True):
    # create functions
    def test_functions(x, noise_variance):
        y_sin = np.sin(2 * x)

        y_square = np.sign(y_sin)
        y_square[y_square == 0] = 1

        if noise_variance is not None:
            std = np.sqrt(noise_variance)
            y_sin += np.random.normal(scale=std, size=y_sin.shape)
            y_square += np.random.normal(scale=std, size=y_square.shape)

        return y_sin, y_square

    x = np.arange(0, 2 * np.pi, 0.1)[:, np.newaxis]
    x_val = np.arange(0.025, 2 * np.pi, 0.1)[:, np.newaxis]
    x_test = np.arange(0.05, 2 * np.pi, 0.1)[:, np.newaxis]

    y_sin, y_square = test_functions(x, noise_variance)
    y_sin_val, y_square_val = test_functions(x_val, noise_variance)
    y_sin_test, y_square_test = test_functions(x_test, noise_variance)

    # plot functions
    if show:
        _, ax = subplots(1, 1, size=figsize(1, 2))

        ax.plot(x, y_sin, label="$sin(2x)$")
        ax.plot(x, y_square, label="$square(2x)$")

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.set_xlim([0, 2 * np.pi])
        ax.set_xticks(np.arange(0, 5 / 2 * np.pi, np.pi / 2))
        ax.set_xticklabels([
            "0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"
        ])

        ax.legend()

    # return datasets
    dataset_sin = GaussianRBFDataset(
        x, y_sin, x_val, y_sin_val, x_test, y_sin_test)

    dataset_square = GaussianRBFDataset(
        x, y_square, x_val, y_square_val, x_test, y_square_test)

    return dataset_sin, dataset_square


def profile_networks(ds,
                     nodes_profile,
                     nodes_preview,
                     kwargs_network=None,
                     kwargs_training=None):

    _, (ax_preview, ax_profile) = subplots(1, 2, size=figsize(1, 3))

    kwargs_network = kwargs_network if kwargs_network else {}
    kwargs_training = kwargs_training if kwargs_training else {}

    training = pop_val(kwargs_training, 'training', DEFAULT_TRAINING)

    def fit_model(nodes, sigma=DEFAULT_NODE_STD):
        model = GaussianRBFNetwork(
            domain=ds.x, nodes=nodes, node_std=sigma, **kwargs_network)

        if training == 'least_squares':
            model.least_squares_fit(ds.x, ds.y)
        elif training == 'delta_rule':
            model.delta_rule_train(ds.x, ds.y, **kwargs_training)

        return model

    # profiling
    std = pop_val(kwargs_network, 'node_std', DEFAULT_NODE_STD)

    for sigma in make_iterable(std):
        residuals = []
        for ns in make_iterable(nodes_profile):
            fmt = "{} units, sigma = {:.2f}"
            sys.stdout.write(fmt.format(ns, sigma).ljust(80) + "\r")
            sys.stdout.flush()

            model = fit_model(ns, sigma)

            residuals.append(model.residual(ds.x_val, ds.y_val))

        if std is not None:
            label = "$\sigma = {:.3f}$".format(sigma)
        else:
            label = None

        ax_profile.semilogy(nodes_profile, residuals,
                            linestyle='--', marker='o',
                            label=label)

    ax_profile.legend()

    for residual_threshold in np.logspace(-3, -1, 3):
        if np.min(residuals) <= residual_threshold:
            ax_profile.axhline(residual_threshold, color='r', linestyle='--')

    ax_profile.set_xlabel("Number of Units")
    ax_profile.set_ylabel("Residual Error (Validation Set)")

    ax_profile.yaxis.set_ticks_position('right')
    ax_profile.yaxis.set_label_position('right')

    # preview
    ax_preview.plot(ds.x_val, ds.y_val, label="Ground Truth")

    for ns in make_iterable(nodes_preview):
        model = fit_model(ns)

        y_pred = model.predict(ds.x)
        res = model.residual(ds.x_val, ds.y_val)

        fmt = "{} Units"
        ax_preview.plot(ds.x, y_pred, label=fmt.format(ns, res))

    ax_preview.set_xlabel("x")
    ax_preview.set_ylabel("y")

    bbox = ax_preview.get_position()
    ax_preview.set_position([bbox.x0, bbox.y0, bbox.width * 0.7, bbox.height])
    ax_preview.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if training == 'least_squares':
        ax_preview.set_title(r"$\sigma = {:.2f}$".format(DEFAULT_NODE_STD))

    elif training == 'delta_rule':
        learning_rate = kwargs_training.get(
            'learning_rate', DEFAULT_LEARNING_RATE_DELTA)

        epochs = kwargs_training.get(
            'epochs', DEFAULT_EPOCHS_DELTA)

        fmt = "$\eta = {:1.2e}$, {} Epochs"
        ax_profile.set_title(fmt.format(learning_rate, epochs))

        fmt = r"$\sigma = {:.2f}$, $\eta = {:1.2e}$, {} Epochs"
        ax_preview.set_title(fmt.format(DEFAULT_NODE_STD, learning_rate, epochs))


def profile_convergence(ds,
                        kwargs_network=None,
                        kwargs_training=None,
                        maximum=10,
                        thresh=None,
                        runs=DEFAULT_RUNS):

    _, ax = subplots(1, 1, size=figsize(1, 2))

    kwargs_network = kwargs_network if kwargs_network else {}
    kwargs_training = kwargs_training if kwargs_training else {}

    node_std = pop_val(kwargs_network, 'node_std', DEFAULT_NODE_STD)

    learning_rate = pop_val(kwargs_training,
                            'learning_rate', DEFAULT_LEARNING_RATE_DELTA)

    epochs = pop_val(kwargs_training, 'epochs', DEFAULT_EPOCHS_DELTA)

    # initialize model
    model = GaussianRBFNetwork(domain=ds.x, **kwargs_network)

    weights_init = model.weights

    error_init = model.residual(ds.x, ds.y)

    # define auxilliary function
    def learning_curve(learning_rate):
        model.weights = weights_init.copy()

        errors = [error_init]

        for epoch in range(1, epochs + 1):
            fmt = "learning rate = {:1.2e}, epoch {}"
            sys.stdout.write(fmt.format(learning_rate, epoch).ljust(80) + "\r")
            sys.stdout.flush()

            model.delta_rule_train(ds.x, ds.y,
                                   learning_rate=learning_rate,
                                   epochs=1,
                                   improve=True,
                                   **kwargs_training)

            error = model.error(ds.x, ds.y)

            if error > maximum:
                errors += [np.nan] * (epochs - len(errors) + 1)
                break

            errors.append(error)

        return errors

    # create learning curves
    cmap = plt.get_cmap('tab10')

    for i, lr in enumerate(make_iterable(learning_rate)):
        if kwargs_network.get('node_pos') == 'random':
            ax.semilogy(range(epochs + 1), learning_curve(lr),
                        color=cmap(i), alpha=ALPHA, linewidth=LINEWIDTH,
                        label=r"$\eta = {:1.2e}$".format(lr))

            for r in range(runs - 1):
                model.init_kernels_random(node_std)

                ax.semilogy(range(epochs + 1), learning_curve(lr),
                            color=cmap(i), alpha=ALPHA, linewidth=LINEWIDTH)
        else:
            ax.semilogy(range(epochs + 1), learning_curve(lr),
                        label=r"$\eta = {:1.2e}$".format(lr))

    if thresh is not None:
        for t in make_iterable(thresh):
            ax.axhline(t, color='r', linestyle='--')

    ax.set_title(r"{} Units, $\sigma = {:.2f}$".format(model.nodes, node_std))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (Training Set)")

    ax.legend()


def profile_rbf_widths(ds,
                       kwargs_network=None,
                       kwargs_training=None):

    _, ax = subplots(1, 1, size=figsize(1, 2))

    ax.plot(ds.x, ds.y, label="Ground Truth")

    kwargs_network = kwargs_network if kwargs_network else {}
    kwargs_training = kwargs_training if kwargs_training else {}

    std = pop_val(kwargs_network, 'node_std', DEFAULT_NODE_STD)
    training = pop_val(kwargs_training, 'training', DEFAULT_TRAINING)

    for sigma in make_iterable(std):
        model = GaussianRBFNetwork(domain=ds.x, node_std=sigma, **kwargs_network)

        if training == 'least_squares':
            model.least_squares_fit(ds.x, ds.y)
        elif training == 'delta_rule':
            model.delta_rule_train(ds.x, ds.y, **kwargs_training)

        y_pred = model.predict(ds.x)

        ax.plot(ds.x, y_pred, label=r"$\sigma = {:.3f}$".format(sigma))

    if training == 'least_squares':
        ax.set_title("{} Units".format(model.nodes))

    elif training == 'delta_rule':
        learning_rate = kwargs_training.get(
            'learning_rate', DEFAULT_LEARNING_RATE_DELTA)

        epochs = kwargs_training.get(
            'epochs', DEFAULT_EPOCHS_DELTA)

        fmt = "{} Units, \eta = {:1.2e}, {} Epochs"
        ax.set_title(fmt.format(model.nodes, learning_rate, epochs))

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.legend()


def profile_clean_performance(datasets_noisy,
                              datasets_clean,
                              kwargs_network=None,
                              kwargs_training=None):

    _, axes = subplots(2, 2, size=figsize(3, 3))

    kwargs_network = kwargs_network if kwargs_network else {}
    kwargs_training = kwargs_training if kwargs_training else {}

    nodes = pop_val(kwargs_network, 'nodes', DEFAULT_NODES)

    it = zip(datasets_noisy, datasets_clean, nodes)
    for i, (ds, ds_clean, nodes_) in enumerate(it):
        for ax in axes[i, :]:
            ax.plot(ds_clean.x_test, ds_clean.y_test, label="Ground Truth")

        def plot_result(model, ns, ax):
            pred = model.predict(ds_clean.x_test)
            error = model.error(ds_clean.x_test, ds_clean.y_test)

            label = "{} Units (MSE is {:1.2e})".format(ns, error)
            ax.plot(ds_clean.x_test, pred, label=label)

        for ns in make_iterable(nodes_):
            sys.stdout.write("{} units ({})".format(ns, i + 1).ljust(80) + "\r")
            sys.stdout.flush()

            model = GaussianRBFNetwork(domain=ds.x, nodes=ns, **kwargs_network)

            model.least_squares_fit(ds.x, ds.y)
            plot_result(model, ns, ax=axes[i, 0])

            model.delta_rule_train(ds.x, ds.y, **kwargs_training)
            plot_result(model, ns, ax=axes[i, 1])

        node_std = kwargs_training.get(
            'node_std', DEFAULT_NODE_STD)

        learning_rate = kwargs_training.get(
            'learning_rate', DEFAULT_LEARNING_RATE_DELTA)

        epochs = kwargs_training.get(
            'epochs', DEFAULT_EPOCHS_DELTA)

        fmt = r"Least Squares Fit ($\sigma$ = {:.2f})"
        axes[i, 0].set_title(fmt.format(node_std))

        fmt = r"Delta Rule ($\sigma$ = {:.2f}, $\eta$ = {:1.2e}, {} Epochs)"
        axes[i, 1].set_title(fmt.format(node_std, learning_rate, epochs))

        for ax in axes[i, :]:
            ax.set_ylim([-3, 2])

            ax.set_xlabel("x")
            ax.set_ylabel("y")

            ax.legend(loc='lower left')


def rbf_network_gridsearch(ds,
                           kwargs_network=None,
                           kwargs_training=None,
                           thresh_abs=0,
                           thresh_div=1,
                           runs=DEFAULT_RUNS,
                           verbose=False):

    kwargs_network = kwargs_network if kwargs_network else {}
    kwargs_training = kwargs_training if kwargs_training else {}

    nodes = pop_val(kwargs_network, 'nodes', DEFAULT_NODES)
    std = pop_val(kwargs_network, 'node_std', DEFAULT_NODE_STD)
    training = pop_val(kwargs_training, 'training', DEFAULT_TRAINING)

    if verbose:
        if training == 'least_squares':
            runs = 1

            fmt_header = "{:<10}" * 3
            fmt_row = ("{:<10d}" + "{:<10.3f}" + "{:<10.2e}")

        elif training == 'delta_rule':
            fmt_header = "{:<10}" * 4
            fmt_row = ("{:<10d}" + "{:<10.3f}" + "{:<10.2e}" * 2)

        header = fmt_header.format("Units", "Std", "E[err]", "std[err]").rstrip()
        print('{}\n{}'.format(header, '-' * len(header)), flush=True)

    best_model = None
    res_min = math.inf

    for ns in make_iterable(nodes):
        for sigma in make_iterable(std):
            if not verbose:
                if best_model is not None:
                    fmt = "{} units, sigma = {:.2f} (best was {})"
                    sys.stdout.write(
                        fmt.format(ns, sigma, best_model[0]).ljust(80) + "\r")
                else:
                    fmt = "{} units, sigma = {:.2f}"
                    sys.stdout.write(
                        fmt.format(ns, sigma).ljust(80) + "\r")

                sys.stdout.flush()

            model = GaussianRBFNetwork(
                nodes=ns, node_std=sigma, domain=ds.x, **kwargs_network)

            if training == 'least_squares':
                model.least_squares_fit(ds.x, ds.y)
                residual = model.residual(ds.x_val, ds.y_val)

                if verbose:
                    print(fmt_row.format(ns, sigma, residual, flush=True))

            elif training == 'delta_rule':
                residuals = []
                for _ in range(runs):
                    model.delta_rule_train(ds.x, ds.y, **kwargs_training)
                    residuals.append(model.residual(ds.x_val, ds.y_val))

                residual = np.mean(residuals)
                res_std = np.std(residuals)

                if verbose:
                    if residual > 1:
                        residual = math.inf
                        res_std = 0

                    print(fmt_row.format(ns, sigma, residual, res_std, flush=True))

            if residual < res_min - thresh_abs and residual < res_min / thresh_div:
                res_min = residual
                best_model = (ns, sigma, res_min)

    return best_model


def rbf_perceptron_compare(ds,
                           rbf_nodes,
                           rbf_node_std,
                           perceptron_nodes,
                           perceptron_learning_rate,
                           perceptron_momentum,
                           perceptron_epochs,
                           runs=DEFAULT_RUNS):

    # RBF network prediction
    model_rbf = GaussianRBFNetwork(domain=ds.x,
                                   nodes=rbf_nodes,
                                   node_std=rbf_node_std)

    times_rbf = []
    errors_rbf = []
    for _ in range(runs):
        start = time()
        model_rbf.least_squares_fit(ds.x, ds.y)
        times_rbf.append(time() - start)

        pred = model_rbf.predict(ds.x_test)
        errors_rbf.append(np.mean((ds.y_test - pred)**2))

    y_pred_rbf = model_rbf.predict(ds.x_test)

    # perceptron prediction
    output_layer = Dense(
        1,
        use_bias=True
    )

    hidden_layer = Dense(
        perceptron_nodes,
        input_shape=(1,),
        activation='sigmoid',
        use_bias=True
    )

    model_perceptron = Sequential([hidden_layer, output_layer])

    sgd = optimizers.SGD(lr=perceptron_learning_rate,
                         momentum=perceptron_momentum)

    model_perceptron.compile(optimizer=sgd, loss='mean_squared_error')

    times_perceptron = []
    errors_perceptron = []
    for _ in range(runs):
        start = time()

        model_perceptron.fit(ds.x,
                             np.ravel(ds.y),
                             batch_size=ds.x.shape[0],
                             shuffle=False,
                             epochs=perceptron_epochs,
                             verbose=0)

        times_perceptron.append(time() - start)

        pred = model_perceptron.predict(ds.x_test)
        errors_perceptron.append(np.mean((ds.y_test - pred)**2))

    y_pred_perceptron = model_perceptron.predict(ds.x_test)

    # print runtimes
    print("RBF training ran in {:1.2e} +/- {:1.2e} seconds".format(
        np.mean(times_rbf), np.std(times_rbf)))

    print("Perceptron training ran in {:1.2e} +/- {:1.2e} seconds".format(
        np.mean(times_perceptron), np.std(times_perceptron)))

    # print errors
    print("RBF MSE is {:1.3e} +/- {:1.3e}".format(
        np.mean(errors_rbf), np.std(errors_rbf)))

    print("Perceptron MSE is {:1.3e} +/- {:1.3e}".format(
        np.mean(errors_perceptron), np.std(errors_perceptron)))

    # plot results
    _, ax = subplots(1, 1, size=figsize(2, 3))

    cmap = plt.get_cmap('tab10')

    res_rbf = abs(y_pred_rbf - ds.y_test)
    res_perceptron = abs(y_pred_perceptron - ds.y_test)

    ax.plot(ds.x_test, ds.y_test, color=cmap(0),
            label="Ground Truth")

    ax.plot(ds.x_test, y_pred_rbf, color=cmap(1),
            label="RBF Network")

    fmt = "RBF Network Residual (MSE is {:.3f})"
    ax.plot(ds.x_test, res_rbf, color=cmap(1), linestyle='--',
            label=fmt.format(np.sum(res_rbf**2) / len(res_rbf)))

    ax.plot(ds.x_test, y_pred_perceptron, color=cmap(2),
            label="Two-Layer Perceptron")

    fmt = "Two-Layer Perceptron Residual (MSE is {:.3f})"
    ax.plot(ds.x_test, res_perceptron, color=cmap(2), linestyle='--',
            label=fmt.format(np.sum(res_perceptron**2) / len(res_perceptron)))

    ax.set_title(r"{} Units, $\sigma$ = {:.2f}".format(rbf_nodes, rbf_node_std))

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_ylim([-2, 2])

    ax.legend()


def rbf_pos_preview(ds,
                    kwargs_network=None):

    _, axes = subplots(1, 3)

    for ax in axes.flatten():
        ax.plot(ds.x, ds.y, label="Input")

    cmap = plt.get_cmap('tab10')

    kwargs_network = kwargs_network if kwargs_network else {}
    node_pos = pop_val(kwargs_network, 'node_pos', DEFAULT_NODE_POS)

    for pos, ax in zip(node_pos, axes.flatten()):
        model = GaussianRBFNetwork(domain=ds.x,
                                   node_pos=pos,
                                   **kwargs_network)

        for j, mu in enumerate(model.mu):
            rbf = np.exp(-(ds.x - mu)**2 / (2 * model.cov[0]))

            ax.plot(ds.x, rbf, color=cmap(1),
                    label=("RBF" if j == 0 else None))

        ax.set_title(pos.capitalize())

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax.legend()


def rbf_pos_compare(ds,
                    kwargs_network=None,
                    kwargs_training=None,
                    runs=DEFAULT_RUNS):

    _, ax = subplots(1, 1, size=figsize(1, 2))

    cmap = plt.get_cmap('tab10')

    kwargs_network = kwargs_network if kwargs_network else {}
    nodes = pop_val(kwargs_network, 'nodes', DEFAULT_NODES)
    node_pos = pop_val(kwargs_network, 'node_pos', DEFAULT_NODE_POS)
    epochs = pop_val(kwargs_training, 'epochs', DEFAULT_EPOCHS_DELTA)

    weights_init = np.random.randn(nodes, 1)

    for i, pos in enumerate(node_pos):
        for r in range(runs):
            fmt = "{} ({})"
            sys.stdout.write(fmt.format(pos, r + 1).ljust(80) + "\r")
            sys.stdout.flush()

            model = GaussianRBFNetwork(domain=ds.x,
                                       nodes=nodes,
                                       node_pos=pos,
                                       **kwargs_network)

            model.weights = weights_init.copy()

            residuals = [model.residual(ds.x_val, ds.y_val)]

            for _ in range(epochs):
                model.delta_rule_train(ds.x,
                                       ds.y,
                                       epochs=1,
                                       improve=True,
                                       **kwargs_training)

                residuals.append(model.residual(ds.x_val, ds.y_val))

            ax.semilogy(range(epochs + 1), residuals,
                        color=cmap(i), alpha=ALPHA, linewidth=LINEWIDTH,
                        label=(pos.capitalize() if r == 0 else None))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Residual Error (Validation Set)")

    ax.legend()
