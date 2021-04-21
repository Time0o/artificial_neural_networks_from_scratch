import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.utils import check_random_state

from .plotting import figsize, subplots


class AutoencoderFactoryKeras(object):
    def __init__(self, inputs, kwargs_model):
        self.inputs = inputs
        self.kwargs_model = kwargs_model

    def __call__(self, bs, lr, reg):
        model = Autoencoder(batch_size=bs,
                            learning_rate=lr,
                            regularization=reg,
                            **self.kwargs_model)

        model._init_weights(self.inputs)
        model._init_model_keras(self.inputs)

        return model._model


class Autoencoder:
    def __init__(self,
                 hidden_nodes,
                 weights_init=None,
                 weight_initializer='random_normal',
                 biases_init=None,
                 activation='sigmoid',
                 batch_size=1,
                 batch_norm=False,
                 learning_rate=1e-3,
                 momentum=0.9,
                 regularization=0,
                 optimizer='adam',
                 max_epochs=sys.maxsize,
                 convergence_criterion=(0, 10),
                 backend='keras'):

        if backend not in ['tensorflow', 'keras', 'sklearn']:
            raise ValueError("invalid backend: {}".format(backend))

        self._hidden_nodes = hidden_nodes
        self._weights_init = weights_init
        self._weight_initializer = weight_initializer
        self._biases_init = biases_init
        self._activation = activation
        self._batch_size = batch_size
        self._batch_norm = batch_norm
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._regularization = regularization
        self._optimizer = optimizer
        self._max_epochs = max_epochs
        self._conv = convergence_criterion
        self._backend = backend

    @staticmethod
    def gridsearch(inputs,
                   batch_sizes,
                   learning_rates,
                   regularizations,
                   kwargs_model=None,
                   cv=10,
                   verbose=False):

        backend = kwargs_model.get('backend', 'keras')

        if backend != 'keras':
            err = "gridsearch currently only implemented for keras backend"
            raise ValueError(err)

        build_fn = AutoencoderFactoryKeras(inputs, kwargs_model)

        epochs = kwargs_model.get('max_epochs', sys.maxsize)

        estimator = KerasRegressor(build_fn=build_fn,
                                   epochs=epochs,
                                   verbose=0)

        search = GridSearchCV(estimator=estimator,
                              scoring='neg_mean_absolute_error',
                              cv=cv,
                              param_grid={
                                  'bs': batch_sizes,
                                  'lr': learning_rates,
                                  'reg': regularizations
                              },
                              return_train_score=True,
                              error_score=np.nan,
                              n_jobs=1,
                              verbose=(51 if verbose else 0))

        conv = kwargs_model.get('convergence_criterion', (0, 10))

        callbacks = [
            EarlyStopping(
                monitor='loss',
                min_delta=conv[0],
                patience=conv[1])
        ]

        return search.fit(inputs, inputs, callbacks=callbacks)

    def train(self,
              inputs,
              inputs_val=None,
              epochs=None,
              learning_curve=False,
              verbose=False):

        kwargs = {
            'inputs_val': inputs_val,
            'epochs': epochs,
            'learning_curve': learning_curve,
            'verbose': verbose
        }

        if self._backend == 'tensorflow':
            self._init_model_tensorflow(inputs)
            return self._train_tensorflow(inputs, **kwargs)
        if self._backend == 'sklearn':
            self._init_model_sklearn(inputs)
            return self._train_sklearn(inputs, **kwargs)
        elif self._backend == 'keras':
            self._init_model_keras(inputs)
            return self._train_keras(inputs, **kwargs)

    def _init_model_tensorflow(self, inputs):
        if self._weights_init is not None or self._biases_init is not None:
            err = "custom weights currently not supported by tf backend"
            raise ValueError(err)

    def _train_tensorflow(self,
                          inputs,
                          inputs_val,
                          epochs,
                          learning_curve,
                          verbose):

        if inputs_val is not None:
            err = "tf backend currently does not support validation"
            raise ValueError(err)

        if learning_curve and learning_curve != 'mse':
            err = "tf backend currently only supports MSE"
            raise ValueError(err)

        # process dataset
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10, inputs.shape[0])

        dataset_it = dataset.make_one_shot_iterator()
        input_layer = dataset_it.get_next()

        # construct hidden and output layer
        layer_settings = {}

        layer_settings['activation'] = {
            'sigmoid': tf.nn.sigmoid,
            'relu': tf.nn.relu,
            'elu': tf.nn.elu
        }[self._activation]

        layer_settings['kernel_initializer'] = {
            'random_normal': tf.initializers.random_normal,
            'xavier': tf.contrib.layers.xavier_initializer(uniform=False),
            'he': tf.contrib.layers.variance_scaling_initializer()
        }[self._weight_initializer]

        if self._regularization > 0:
            layer_settings['kernel_regularizer'] = \
                tf.contrib.layers.l2_regularizer(self._regularization)

        def layer(prev, nodes):
            print(layer_settings) # TODO
            res = tf.layers.dense(prev, nodes, **layer_settings)

            if self._batch_norm:
                res = tf.layers.batch_normalization(
                    res, training=True, momentum=0.9)

            return res

        hidden_layer = layer(input_layer, self._hidden_nodes)
        output_layer = layer(hidden_layer, inputs.shape[1])

        # set up otimization
        optimizer = {
            'momentum': tf.train.MomentumOptimizer(
                self._learning_rate, self._momentum),
            'momentum_nesterov': tf.train.MomentumOptimizer(
                self._learning_rate, self._momentum, use_nesterov=True),
            'adam': tf.train.AdamOptimizer(self._learning_rate)
        }[self._optimizer]

        loss = tf.reduce_mean(tf.square(output_layer - input_layer))
        training_op = optimizer.minimize(loss)

        # train model
        errors = []

        batches = inputs.shape[0] // self._batch_size

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if epochs is None:
                err = "early stopping currently not supported by tf backend"
                raise ValueError(err)

            for e in range(epochs):
                for _ in range(batches):
                    _, loss_ = sess.run([training_op, loss])

                # determine current error
                errors.append(loss_)

                if verbose:
                    self._show_progress(e, epochs)

        # return learning curve
        if learning_curve:
            epochs = list(range(1, len(errors) + 1))

            return epochs, errors

    def _init_model_keras(self, inputs):
        # construct network
        def initializer(weights):
            def res(shape, dtype=None):
                assert shape == res.weights.shape

                if dtype is not None:
                    weights = res.weights.astype(dtype)
                else:
                    weights = res.weights

                return weights

            res.weights = weights

            return res

        if self._weights_init is not None:
            kernel_init_hidden = initializer(self._weights_init[0])
            kernel_init_output = initializer(self._weights_init[1])
        else:
            kernel_init_hidden = kernel_init_output = {
                'random_normal': 'RandomNormal',
                'xavier': 'glorot_normal',
                'he': 'he_normal'
            }[self._weight_initializer]

        if self._biases_init is not None:
            bias_init_hidden = initializer(self._biases_init[0])
            bias_init_output = initializer(self._biases_init[1])
        else:
            bias_init_hidden = 'Zeros'
            bias_init_output = 'Zeros'

        hidden_layer = Dense(
            self._hidden_nodes,
            input_shape=(inputs.shape[1],),
            activation=self._activation,
            kernel_initializer=kernel_init_hidden,
            bias_initializer=bias_init_hidden,
            kernel_regularizer=l2(self._regularization),
        )

        output_layer = Dense(
            inputs.shape[1],
            activation=self._activation,
            kernel_initializer=kernel_init_output,
            bias_initializer=bias_init_output,
            kernel_regularizer=l2(self._regularization)
        )

        self._model = Sequential([hidden_layer, output_layer])

        # set up optimization
        opt = {
            'momentum': SGD(
                lr=self._learning_rate, momentum=self._momentum),
            'momentum_nesterov': SGD(
                lr=self._learning_rate, momentum=self._momentum, nesterov=True),
            'adam': Adam(lr=self._learning_rate)
        }[self._optimizer]

        self._model.compile(optimizer=opt, loss='mean_squared_error')

    def _train_keras(self,
                     inputs,
                     inputs_val,
                     epochs,
                     learning_curve,
                     verbose):

        # compute initial errors
        if learning_curve:
            errors = []

            if inputs_val is not None:
                errors_val = []

        # define convergence criterion
        if epochs is None:
            callbacks = [
                EarlyStopping(
                    monitor='loss',
                    min_delta=self._conv[0],
                    patience=self._conv[1],
                    verbose=(1 if verbose else 0))
            ]

            epochs = self._max_epochs
        else:
            callbacks = []

        # set up validation set
        if inputs_val is not None:
            validation_data = (inputs_val, inputs_val)
        else:
            validation_data = None

        # train model
        for e in range(epochs):
            h = self._model.fit(inputs,
                                inputs,
                                validation_data=validation_data,
                                batch_size=self._batch_size,
                                epochs=(e + 1),
                                initial_epoch=e,
                                callbacks=callbacks,
                                verbose=0)
                
            # determine current error
            if learning_curve:
                if learning_curve == 'mse':
                    errors.append(h.history['loss'][0])

                    if inputs_val is not None:
                        errors_val.append(h.history['val_loss'][0])

                elif learning_curve == 'total':
                    errors.append(self.error(inputs))

                    if inputs_val is not None:
                        errors_val.append(self.error(inputs_val))

            # show progress
            if verbose:
                self._show_progress(e, epochs)

        # return learning curve
        if learning_curve:
            epochs = list(range(1, len(errors) + 1))

            if inputs_val is not None:
                return epochs, errors, errors_val
            else:
                return epochs, errors

    def _init_model_sklearn(self, inputs):
        self._model = MLPRegressor(
            # structure
            hidden_layer_sizes=(self._hidden_nodes,),
            # activation functions
            activation='logistic',
            # solver
            solver='sgd',
            warm_start=True,
            # batch size
            batch_size=self._batch_size,
            # learning rate
            learning_rate='constant',
            learning_rate_init=self._learning_rate,
            # momentum
            momentum=self._momentum,
            nesterovs_momentum=True,
            # regularization
            alpha=self._regularization,
            # convergence
            max_iter=self._max_epochs,
            tol=self._conv[0],
            n_iter_no_change=self._conv[1])

    def _train_sklearn(self,
                       inputs,
                       inputs_val,
                       epochs,
                       learning_curve,
                       verbose):

        if learning_curve and learning_curve != 'total':
            err = "sklearn backend currently only supports total error"
            raise ValueError(err)

        # initialize weights and biases
        if self._weights_init is None:
            self._weights_init = [
                np.random.randn(inputs.shape[1], self._hidden_nodes),
                np.random.randn(self._hidden_nodes, inputs.shape[1])
            ]

        if self._biases_init is None:
            self._biases_init = [
                np.zeros(self._hidden_nodes),
                np.zeros(inputs.shape[1])
            ]

        # initialize learning curve
        if learning_curve:
            total_errors = []

            if inputs_val is not None:
                total_errors_val = []

            best_total_error = math.inf
            dead_epochs = 0

        epoch = 0
        while True:
            if epoch == 0:
                # hack ahead, scikit learn's awful MLPRegressor interface
                # ordinarily does not allow manual weight initialization

                self._model.n_outputs_ = inputs.shape[1]

                self._model._random_state = check_random_state(
                    self._model.random_state)

                self._model._initialize(
                    inputs,
                    [inputs.shape[1], self._hidden_nodes, inputs.shape[1]])

                self._model.coefs_ = self._weights_init
                self._model.intercepts_ = self._biases_init

                continue
            else:
                self._model = self._model.partial_fit(inputs, inputs)

            # determine current error
            total_error = self.error(inputs)

            if learning_curve:
                total_errors.append(total_error)

                if inputs_val is not None:
                    total_errors_val.append(self.error(inputs_val))

            # show progress
            if verbose:
                self._show_progress(epoch, epochs)

            # check for convergence
            epoch += 1

            if epochs is None:
                if total_error >= best_total_error - sel._conv[0]:
                    dead_epochs += 1
                    if dead_epochs == self._conv[1]:
                        break

                if total_error < best_total_error:
                    best_total_error = min(best_total_error, total_error)
                    dead_epochs = 0
            else:
                if epoch > epochs:
                    break

        # return learning curve
        if learning_curve:
            total_epochs = list(range(1, len(errors) + 1))

            if inputs_val is not None:
                return total_epochs, total_errors, total_errors_val
            else:
                return total_epochs, total_errors

    def predict(self, i):
        return self._model.predict(i.reshape(1, len(i)))

    def error(self, inputs):
        total_error = 0
        for i in inputs:
            pred = self.predict(i)
            total_error += np.mean(np.abs(pred - i))

        return total_error

    @staticmethod
    def _show_progress(e, epochs):
        if epochs is None:
            print("\repoch {}".format(e))

        else:
            bar = '=' * int(50 * (e + 1) / epochs)
            progress = "[{:<50}] epoch {}/{}".format(bar, e + 1, epochs)

            print("\r" + progress, end='')


def learning_curve(model,
                   inputs,
                   inputs_val=None,
                   train_args=None,
                   metric='mse',
                   ax=None,
                   log=True,
                   labels=None):

    if ax is None:
        _, ax = subplots(1, 1, size=figsize(1, 2))

    if train_args is None:
        train_args = {}

    res = model.train(inputs,
                      learning_curve=metric,
                      **train_args)

    plt = ax.semilogy if log else ax.plot

    if labels is None:
        labels = ["Training Set", "Validation Set"]

    p = plt(res[0], res[1],
            label=labels[0])

    if inputs_val is not None:
        plt(res[0], res[2],
            color=p[0].get_color(),
            linestyle='--',
            label=labels[1])

    ax.set_xlabel("Epoch")

    metric_name = {
        'mse': 'MSE',
        'total': 'Total Error',
    }[metric]

    ax.set_ylabel(metric_name)

    ax.legend()


def sample_learning_curves(model_args,
                           inputs,
                           train_args=None,
                           param='learning_rate',
                           symbol="$\\eta$",
                           metric='mse',
                           ax=None,
                           log=True):

    if ax is None:
        _, ax = subplots(1, 1, size=figsize(1, 2))

    for val in model_args[param]:
        model_args_ = model_args.copy()
        model_args_[param] = val

        model = Autoencoder(**model_args_)

        learning_curve(inputs=inputs,
                       model=model,
                       train_args=train_args,
                       metric=metric,
                       ax=ax,
                       log=log,
                       labels=["{} = {}".format(symbol, val)])


def compare_activations(inputs,
                        model_args,
                        train_args,
                        ax=None,
                        labels=None):

    if ax is None:
        _, ax = subplots(1, 1, size=figsize(1, 2))

    model_args = model_args.copy()

    weight_initializers = model_args.pop('weight_initializer')
    activations = model_args.pop('activation')
    learning_rates = model_args.pop('learning_rate')

    for wi, a, lr, label in zip(weight_initializers,
                                activations,
                                learning_rates,
                                labels):

        model = Autoencoder(weight_initializer=wi,
                            activation=a,
                            learning_rate=lr,
                            **model_args)

        learning_curve(model=model,
                       inputs=inputs,
                       train_args=train_args,
                       ax=ax,
                       labels=[label])
