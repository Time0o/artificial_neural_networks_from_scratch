import os; os.environ['KERAS_BACKEND'] = 'theano'
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2

from .plotting import figsize, subplots
from .util import make_iterable


DEFAULT_ACTIVATE_OUTPUT = False
DEFAULT_HIDDEN_NODES = 2
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_LEARNING_MOMENTUM = 0.9
DEFAULT_REGULARIZATION = 0.0001
DEFAULT_STOP_EARLY = True
DEFAULT_STOP_TOLERANCE = 0.001
DEFAULT_STOP_ITERATIONS = 10
DEFAULT_EPOCHS = 100
DEFAULT_BACKEND = 'keras'
DEFAULT_TRAINING_NOISE = None
DEFAULT_RUNS = 20


class TimeSeriesDataset:
    def __init__(self, t, inputs, labels, n_train, n_val, noise=None):
        self.t = t

        self.n_train = n_train
        self.n_val = n_val
        self.n_test = inputs.shape[1] - n_train - n_val

        self.noise = noise

        train, val, test = self._training_split(inputs, labels)

        self.patterns_train, self.targets_train = train
        self.patterns_val, self.targets_val = val
        self.patterns_test, self.targets_test = test

        self.n_inputs = self.patterns_train.shape[0]
        self.n_outputs = self.targets_train.shape[0]

    def preview(self, ax=None):
        if ax is None:
            _, ax = subplots(1, 1, size=figsize(1, 2))

            ax.set_title("Training/Validation/Test Split")
            ax.set_xlabel("t")
            ax.set_ylabel("x(t)")

        sep1 = self.n_train
        sep2 = sep1 + self.n_val
        sep3 = sep2 + self.n_test

        ax.plot(self.t[:sep1],
                np.ravel(self.targets_train),
                label="Training Data")

        ax.plot(self.t[sep1:sep2],
                np.ravel(self.targets_val),
                label="Validation Data")

        ax.plot(self.t[sep2:sep3],
                np.ravel(self.targets_test),
                label="Test Data")

        for sep in sep1, sep2:
            ax.axvline(self.t[0] + sep, color='k', linestyle='--')

        ax.legend()

    def _training_split(self, inputs, labels):
        def split(p, t, start, n):
             return p[:, start:(start + n)], t[start:(start + n)].reshape(1, n)

        train = split(inputs, labels, 0, self.n_train)
        val = split(inputs, labels, self.n_train, self.n_val)
        test = split(inputs, labels, self.n_train + self.n_val, self.n_test)

        return train, val, test


class TimeSeriesPredictor:
    def __init__(self,
                 inputs,
                 outputs,
                 activate_output=DEFAULT_ACTIVATE_OUTPUT,
                 hidden_nodes=DEFAULT_HIDDEN_NODES,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 learning_momentum=DEFAULT_LEARNING_MOMENTUM,
                 regularization=DEFAULT_REGULARIZATION,
                 stop_early=DEFAULT_STOP_EARLY,
                 stop_tolerance=DEFAULT_STOP_TOLERANCE,
                 stop_iterations=DEFAULT_STOP_ITERATIONS,
                 backend=DEFAULT_BACKEND):

        if backend not in ['keras', 'sklearn']:
            raise ValueError("invalid backend")

        self.backend = backend

        self.inputs = inputs
        self.outputs = outputs
        self.activate_output = activate_output
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.learning_momentum = learning_momentum
        self.regularization = regularization
        self.stop_tolerance = stop_tolerance
        self.stop_iterations = stop_iterations
        self.stop_early = stop_early

        self._trained = False

    def train(self,
              ds,
              epochs=DEFAULT_EPOCHS,
              noise=None,
              final=False,
              verbose=False):

        patterns_train = ds.patterns_train.copy()
        targets_train = ds.targets_train.copy()

        patterns_val = ds.patterns_val.copy()
        targets_val = ds.targets_val.copy()

        if final:
            inputs = np.hstack((patterns_train, patterns_val)).T
            labels = np.hstack((targets_train, targets_val)).T

        else:
            inputs = patterns_train.T
            labels = targets_train.T

            inputs_val = patterns_val.T

        if noise is not None and noise > 0:
            inputs += noise * np.random.randn(*inputs.shape)

        if not final and self.stop_early:
            stop_early_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=self.stop_tolerance,
                patience=self.stop_iterations,
                verbose=verbose
            )

            callbacks=[stop_early_callback]
            validation_data=[inputs_val, np.ravel(targets_val)]

        else:
            callbacks = None
            validation_data = None

        self._reset_model()

        self.model.fit(inputs,
                       np.ravel(labels),
                       batch_size=ds.n_train,
                       callbacks=callbacks,
                       validation_data=validation_data,
                       epochs=epochs,
                       verbose=0)

        self._trained = True

        if not final and self.stop_early:
            return stop_early_callback.stopped_epoch

    def predict(self, ds):
        if not self._trained:
            raise ValueError("model has not been trained yet")

        predictions = [
            self.model.predict(p.T)
            for p in (ds.patterns_train, ds.patterns_val, ds.patterns_test)
        ]

        if self.backend == 'keras':
            return [p.T for p in predictions]
        else:
            return [p[np.newaxis, :] for p in predictions]

    def training_error(self, ds):
        predicted, _, _ = self.predict(ds)

        return self._error(predicted, ds.targets_train)

    def validation_error(self, ds):
        _, predicted, _ = self.predict(ds)

        return self._error(predicted, ds.targets_val)

    def test_error(self, ds):
        _, _, predicted = self.predict(ds)

        return self._error(predicted, ds.targets_test)

    def weights(self):
        w = []
        for hl in self.hidden_layers:
            w1, w2 = hl.get_weights()
            w += [w1.flatten(), w2]

        w1, w2 = self.output_layer.get_weights()
        w += [w1.flatten(), w2]

        return np.concatenate(w)

    def preview(self, ds, ax=None):
        if ax is None:
            _, ax = subplots(1, 1, size=figsize(1, 2))

            ax.set_title("Model Predictions")
            ax.set_xlabel("t")
            ax.set_ylabel("x(t)")

        target = np.hstack((ds.targets_train, ds.targets_val, ds.targets_test))

        ax.plot(ds.t, np.ravel(target),
                label="Data")

        ax.plot(ds.t, np.ravel(np.hstack(self.predict(ds))), color='r',
                label="Prediction")

        ax.legend()

    def _error(self, predicted, targets):
        return np.sum((predicted - np.ravel(targets))**2) / predicted.shape[1]

    def _reset_model(self):
        if self.activate_output:
            self.output_layer = Dense(
                self.outputs,
                activation='sigmoid',
                use_bias=True,
                kernel_regularizer=l2(self.regularization)
            )

        else:
            self.output_layer = Dense(
                self.outputs,
                use_bias=True,
                kernel_regularizer=l2(self.regularization)
            )

        if isinstance(self.hidden_nodes, tuple):
            hidden_layer1 = Dense(
                self.hidden_nodes[0],
                input_shape=(self.inputs,),
                activation='sigmoid',
                use_bias=True,
                kernel_regularizer=l2(self.regularization)
            )

            hidden_layer2 = Dense(
                self.hidden_nodes[1],
                activation='sigmoid',
                use_bias=True,
                kernel_regularizer=l2(self.regularization)
            )

            self.hidden_layers = [hidden_layer1, hidden_layer2]

            self.model = Sequential([hidden_layer1,
                                     hidden_layer2,
                                     self.output_layer])
        else:
            hidden_layer = Dense(
                self.hidden_nodes,
                input_shape=(self.inputs,),
                activation='sigmoid',
                use_bias=True,
                kernel_regularizer=l2(self.regularization)
            )

            self.hidden_layers = [hidden_layer]

            self.model = Sequential([hidden_layer,
                                     self.output_layer])

        sgd = optimizers.SGD(lr=self.learning_rate,
                             momentum=self.learning_momentum)

        self.model.compile(optimizer=sgd, loss='mean_squared_error')


def mackey_glass(t, beta=0.2, gamma=0.1, n=10, theta=25):
    x = np.empty(t[-1] + 1)

    x[0] = 1.5

    for t_ in range(1, t[-1] + 1):
        x_1 = x[t_ - 1]
        x_25 = 0 if t_ < theta else x[t_ - theta]

        x[t_] = x_1 + (beta * x_25) / (1 + x_25**n) - gamma * x_1

    return x


def test_dataset(noise=None):
    t = np.arange(301, 1501)
    x = mackey_glass(np.arange(1506), theta=25)

    patterns = np.array([x[t - offs] for offs in [20, 15, 10, 5, 0]])
    targets = x[t + 5]

    if noise is not None and noise > 0:
        targets += noise * np.random.randn(*targets.shape)

    n_train = 800
    n_val = 200

    return TimeSeriesDataset(t, patterns, targets, n_train, n_val, noise)


def evaluate_networks(dataset,
                      hidden_nodes=DEFAULT_HIDDEN_NODES,
                      learning_rate=DEFAULT_LEARNING_RATE,
                      learning_momentum=DEFAULT_LEARNING_MOMENTUM,
                      regularization=DEFAULT_REGULARIZATION,
                      stop_early=DEFAULT_STOP_EARLY,
                      stop_tolerance=DEFAULT_STOP_TOLERANCE,
                      stop_iterations=DEFAULT_STOP_ITERATIONS,
                      epochs=DEFAULT_EPOCHS,
                      backend=DEFAULT_BACKEND,
                      training_noise=DEFAULT_TRAINING_NOISE,
                      runs=DEFAULT_RUNS,
                      ax=None):

    if ax is None:
        _, ax = subplots(1, 1, size=figsize(2, 2))

    for reg in make_iterable(regularization):
        print("profiling: alpha = {}".format(reg))

        errors = []
        for hn in make_iterable(hidden_nodes):
            model = TimeSeriesPredictor(inputs=5,
                                        outputs=1,
                                        hidden_nodes=hn,
                                        learning_rate=learning_rate,
                                        learning_momentum=learning_momentum,
                                        regularization=reg,
                                        stop_early=stop_early,
                                        stop_tolerance=stop_tolerance,
                                        stop_iterations=stop_iterations,
                                        backend=backend)

            errors_ = []
            for _ in range(runs):
                model.train(dataset, epochs=epochs, noise=training_noise)
                errors_.append(model.validation_error(dataset))

            errors.append(errors_)

        err_mean = [np.mean(e) for e in errors]
        err_std = [np.std(e) for e in errors]
        ax.errorbar(hidden_nodes, err_mean, yerr=err_std, capsize=5,
                    label=r"$\alpha = {}$".format(reg))

    ax.set_xlabel("Hidden Nodes")
    ax.set_ylabel("MSE")

    ax.legend()
    ax.grid()


def evaluate_networks_table(dataset,
                            hidden_nodes=DEFAULT_HIDDEN_NODES,
                            learning_rate=DEFAULT_LEARNING_RATE,
                            learning_momentum=DEFAULT_LEARNING_MOMENTUM,
                            regularization=DEFAULT_REGULARIZATION,
                            stop_early=DEFAULT_STOP_EARLY,
                            stop_tolerance=DEFAULT_STOP_TOLERANCE,
                            stop_iterations=DEFAULT_STOP_ITERATIONS,
                            epochs=DEFAULT_EPOCHS,
                            backend=DEFAULT_BACKEND,
                            training_noise=DEFAULT_TRAINING_NOISE,
                            runs=DEFAULT_RUNS):

    print(("{:<5}" + "{:<12}" * 9).format(
        "HN", "REG",
        "ETR (Mean)", "ETR (Std)",
        "EV (Mean)", "EV (Std)",
        "ET (Mean)", "ET (Std)",
        "EP (Mean)", "EP (Std)"))

    sys.stdout.flush()

    for hn in make_iterable(hidden_nodes):
        for reg in make_iterable(regularization):
            model = TimeSeriesPredictor(inputs=5,
                                        outputs=1,
                                        hidden_nodes=hn,
                                        learning_rate=learning_rate,
                                        learning_momentum=learning_momentum,
                                        regularization=reg,
                                        stop_early=stop_early,
                                        stop_tolerance=stop_tolerance,
                                        stop_iterations=stop_iterations,
                                        backend=backend)

            train_errors = []
            val_errors = []
            test_errors = []
            stopped_epochs = []

            for _ in range(runs):
                epoch = model.train(dataset,
                                    epochs=epochs,
                                    noise=training_noise)

                train_errors.append(model.training_error(dataset))
                val_errors.append(model.validation_error(dataset))
                test_errors.append(model.test_error(dataset))

                if stop_early and epoch > 0:
                    stopped_epochs.append(epoch)
                else:
                    stopped_epochs.append(epochs)

            print(("{:<5}" + "{:<12.2e}" * 7 + "{:<12.2f}" * 2).format(
                hn[1] if isinstance(hn, tuple) else hn, reg,
                np.mean(train_errors), np.std(train_errors),
                np.mean(val_errors), np.std(val_errors),
                np.mean(test_errors), np.std(test_errors),
                np.mean(stopped_epochs), np.std(stopped_epochs)))

            sys.stdout.flush()


def evaluate_weight_distribution(dataset,
                                 hidden_nodes=DEFAULT_HIDDEN_NODES,
                                 learning_rate=DEFAULT_LEARNING_RATE,
                                 learning_momentum=DEFAULT_LEARNING_MOMENTUM,
                                 regularization=DEFAULT_REGULARIZATION,
                                 stop_early=DEFAULT_STOP_EARLY,
                                 stop_tolerance=DEFAULT_STOP_TOLERANCE,
                                 stop_iterations=DEFAULT_STOP_ITERATIONS,
                                 epochs=DEFAULT_EPOCHS,
                                 backend=DEFAULT_BACKEND,
                                 training_noise=DEFAULT_TRAINING_NOISE,
                                 runs=DEFAULT_RUNS,
                                 ax=None):

    if ax is None:
        _, ax = subplots(1, 1, size=figsize(2, 2))

    for reg in make_iterable(regularization):
        model = TimeSeriesPredictor(inputs=5,
                                    outputs=1,
                                    hidden_nodes=hidden_nodes,
                                    learning_rate=learning_rate,
                                    learning_momentum=learning_momentum,
                                    regularization=reg,
                                    stop_early=stop_early,
                                    stop_tolerance=stop_tolerance,
                                    stop_iterations=stop_iterations,
                                    backend=backend)

        weights = []
        for _ in range(runs):
            model.train(dataset, epochs=epochs, noise=training_noise)
            weights.append(model.weights())

        ax.hist(np.concatenate(weights), alpha=0.5,
                label=r"$\alpha = {}$".format(reg))

    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Frequency")

    ax.legend()


def evaluate_generalization(dataset,
                            hidden_nodes=DEFAULT_HIDDEN_NODES,
                            learning_rate=DEFAULT_LEARNING_RATE,
                            learning_momentum=DEFAULT_LEARNING_MOMENTUM,
                            regularization=DEFAULT_REGULARIZATION,
                            stop_early=DEFAULT_STOP_EARLY,
                            stop_tolerance=DEFAULT_STOP_TOLERANCE,
                            stop_iterations=DEFAULT_STOP_ITERATIONS,
                            epochs=DEFAULT_EPOCHS,
                            final=False,
                            backend=DEFAULT_BACKEND,
                            training_noise=DEFAULT_TRAINING_NOISE,
                            runs=DEFAULT_RUNS,
                            ax=None):
    if ax is None:
        _, ax = subplots(1, 1)

    cmap = plt.get_cmap('tab10')

    model = TimeSeriesPredictor(inputs=5,
                                outputs=1,
                                hidden_nodes=hidden_nodes,
                                learning_rate=learning_rate,
                                learning_momentum=learning_momentum,
                                regularization=regularization,
                                stop_early=stop_early,
                                stop_tolerance=stop_tolerance,
                                stop_iterations=stop_iterations,
                                backend=backend)

    t = dataset.t[-dataset.n_test:]

    ax.plot(t, dataset.targets_test.flatten() - np.mean(dataset.targets_test),
            color=cmap(0), label="Target (Standardized)")

    errors = []
    for r in range(runs):
        model.train(dataset, epochs=epochs, noise=training_noise, final=final)

        _, _, prediction = model.predict(dataset)

        errors.append(model.test_error(dataset))

        pred = np.ravel(prediction.flatten() - np.mean(prediction))
        delta = np.ravel(np.abs(dataset.targets_test - prediction))

        if r == 0:
            ax.plot(t, pred,
                    color=cmap(1), label="Prediction (Standardized)")

            if runs == 1:
                ax.plot(t, delta,
                        color=cmap(2), label="|Target - Prediction|")

        else:
            ax.plot(t, pred, color=cmap(1))

    if runs > 1:
        ax.set_title("MSE = {:1.3e} +/- {:1.3e}".format(
            np.mean(errors), np.std(errors)))
    else:
        ax.set_title("MSE = {:1.3e}".format(errors[0]))

    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")

    ax.legend()


def evaluate_noise1(datasets,
                    hidden_nodes=DEFAULT_HIDDEN_NODES,
                    learning_rate=DEFAULT_LEARNING_RATE,
                    learning_momentum=DEFAULT_LEARNING_MOMENTUM,
                    regularization=DEFAULT_REGULARIZATION,
                    stop_early=DEFAULT_STOP_EARLY,
                    stop_tolerance=DEFAULT_STOP_TOLERANCE,
                    stop_iterations=DEFAULT_STOP_ITERATIONS,
                    epochs=DEFAULT_EPOCHS,
                    backend=DEFAULT_BACKEND,
                    runs=DEFAULT_RUNS,
                    ax=None):

    if ax is None:
        _, ax = subplots(1, 1)

    for dataset in datasets:
        errors = []

        for hn in make_iterable(hidden_nodes):
            model = TimeSeriesPredictor(inputs=5,
                                        outputs=1,
                                        hidden_nodes=hn,
                                        learning_rate=learning_rate,
                                        learning_momentum=learning_momentum,
                                        regularization=regularization,
                                        stop_early=stop_early,
                                        stop_tolerance=stop_tolerance,
                                        stop_iterations=stop_iterations,
                                        backend=backend)

            errors_ = []

            for _ in range(runs):
                model.train(dataset, epochs=epochs)

                errors_.append(model.validation_error(dataset))

            errors.append(errors_)

        ax.errorbar([hn2 for hn1, hn2 in hidden_nodes],
                    [np.mean(e) for e in errors],
                    yerr=[np.std(e) for e in errors],
                    capsize=5,
                    label=r"$\sigma = {}$".format(dataset.noise))

        ax.set_xlabel("Hidden Nodes in Second Layer")
        ax.set_ylabel("MSE")

        ax.legend()
        ax.grid()


def evaluate_noise2(datasets,
                    hidden_nodes=DEFAULT_HIDDEN_NODES,
                    learning_rate=DEFAULT_LEARNING_RATE,
                    learning_momentum=DEFAULT_LEARNING_MOMENTUM,
                    regularization=DEFAULT_REGULARIZATION,
                    stop_early=DEFAULT_STOP_EARLY,
                    stop_tolerance=DEFAULT_STOP_TOLERANCE,
                    stop_iterations=DEFAULT_STOP_ITERATIONS,
                    epochs=DEFAULT_EPOCHS,
                    backend=DEFAULT_BACKEND,
                    runs=DEFAULT_RUNS,
                    ax=None):

    if ax is None:
        _, ax = subplots(1, 1)

    for dataset in make_iterable(datasets):
        train_errors = []
        val_errors = []

        for reg in make_iterable(regularization):
            model = TimeSeriesPredictor(inputs=5,
                                        outputs=1,
                                        hidden_nodes=hidden_nodes,
                                        learning_rate=learning_rate,
                                        regularization=reg,
                                        stop_early=stop_early,
                                        stop_tolerance=stop_tolerance,
                                        stop_iterations=stop_iterations)

            train_errors_ = []
            val_errors_ = []

            for _ in range(runs):
                model.train(dataset, epochs=epochs)

                train_errors_.append(model.training_error(dataset))
                val_errors_.append(model.validation_error(dataset))

            train_errors.append(train_errors_)
            val_errors.append(val_errors_)

        ax.errorbar(regularization,
                    [np.mean(e) for e in val_errors],
                    yerr=[np.std(e) for e in val_errors],
                    linestyle='--', capsize=5,
                    label=r"$\sigma = {}$".format(dataset.noise))

        ax.set_xscale('log')

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$MSE$")

    ax.legend()
    ax.grid()


def evaluate_training_time(dataset,
                           hidden_nodes=DEFAULT_HIDDEN_NODES,
                           learning_rate=DEFAULT_LEARNING_RATE,
                           learning_momentum=DEFAULT_LEARNING_MOMENTUM,
                           epochs=DEFAULT_EPOCHS,
                           backend=DEFAULT_BACKEND,
                           runs=DEFAULT_RUNS,
                           ax=None):

    if ax is None:
        _, ax = subplots(1, 1)

    times = []

    for hn in make_iterable(hidden_nodes):
        model = TimeSeriesPredictor(inputs=5,
                                    outputs=1,
                                    hidden_nodes=hn,
                                    learning_rate=learning_rate,
                                    learning_momentum=learning_momentum,
                                    regularization=0,
                                    stop_early=False,
                                    backend=backend)

        times_ = []

        for _ in range(runs):
            start = time()

            model.train(dataset, epochs=epochs)

            times_.append(time() - start)

        times.append(times_)

    if isinstance(hidden_nodes[0], tuple):
        hidden_nodes = [h[0] for h in hidden_nodes]

    ax.errorbar(hidden_nodes,
                [np.mean(t) for t in times],
                yerr=[np.std(t) for t in times])

    ax.set_xlabel("Hidden Nodes")
    ax.set_ylabel("Execution Time (Seconds)")
