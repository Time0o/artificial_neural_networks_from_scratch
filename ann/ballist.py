import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from matplotlib.patches import Circle, Ellipse
from scipy.stats import multivariate_normal

from .plotting import figure, figsize, subplots
from .rbf_network import GaussianRBFNetwork, \
                         DEFAULT_NODES, \
                         DEFAULT_NODE_STD, \
                         DEFAULT_CL_SYMMETRICAL, \
                         DEFAULT_TRAINING, \
                         DEFAULT_RUNS
from .util import make_iterable, pop_val


class BestModel:
    def __init__(self, nodes, std, symmetrical, mse):
        self.nodes = nodes
        self.node_std = std
        self.symmetrical = symmetrical
        self.mse = mse


def ballist_load(f):
    return np.hsplit(np.genfromtxt(f), 4)


def ballist_plot_units(model, angles, velocities):
    _, ax = subplots(1, 1, size=figsize(2, 2))

    ax.scatter(angles, velocities)

    for mu, cov in zip(model.mu, model.cov):
        ax.scatter(*mu, marker='x', color='r')

        vals, vects = la.eig(cov)

        if vals[0] >= vals[1]:
            h = np.sqrt(vals[0])
            w = np.sqrt(vals[1])
            rot = vects[:, 0]
        else:
            h = np.sqrt(vals[1])
            w = np.sqrt(vals[0])
            rot = vects[:, 1]

        s = -2 * np.log(1 - 0.682)
        a = np.arccos(rot[0] / la.norm(rot)) / np.pi * 90

        c = Circle(mu, radius=np.sqrt(max(cov[0, 0], cov[1, 1])),
                   color='g', fill=False)

        #e = Ellipse(mu, width=s*w, height=s*h, angle=a,
        #            color='r', fill=False)

        ax.add_artist(c)
        #ax.add_artist(e)

    ax.set_xlabel("Angle")
    ax.set_ylabel("Velocity")


def ballist_gridsearch(inputs_train,
                       outputs_train,
                       inputs_val,
                       outputs_val,
                       kwargs_network=None,
                       kwargs_training=None,
                       runs=DEFAULT_RUNS,
                       verbose=False,
                       colwidth=15):

    kwargs_network = kwargs_network if kwargs_network else {}
    kwargs_training = kwargs_training if kwargs_training else {}

    if verbose:
        cols = ["Units", "Sigma", "Symmetrical", "MSE (mean)", "MSE (std)"]
        fmt = "{:<{w}}" * 5

        header = fmt.format(*cols, w=colwidth).rstrip()
        print("{}\n{}".format(header, '-' * (len(header))), flush=True)

    best_model = None
    mse_min = math.inf

    nodes = pop_val(kwargs_network, 'nodes', DEFAULT_NODES)
    node_std = pop_val(kwargs_network, 'node_std', DEFAULT_NODE_STD)
    sym = pop_val(kwargs_network, 'cl_symmetrical', DEFAULT_CL_SYMMETRICAL)
    training = pop_val(kwargs_training, 'training', DEFAULT_TRAINING)

    for ns in make_iterable(nodes):
        for sigma in make_iterable(node_std):
            for symmetrical in make_iterable(sym):
                mses = []

                for _ in range(runs):
                    model = GaussianRBFNetwork(domain=inputs_train,
                                               nodes=ns,
                                               node_std=sigma,
                                               cl_symmetrical=symmetrical,
                                               **kwargs_network)

                    if training == 'least_squares':
                        model.least_squares_fit(inputs_train,
                                                outputs_train)

                    elif training == 'delta_rule':
                        model.delta_rule_train(inputs_train,
                                               outputs_train,
                                               **kwargs_training)

                    outputs_pred = model.predict(inputs_val)

                    mse = np.sum((outputs_pred - outputs_val)**2)
                    mse /= outputs_val.shape[0]

                    mses.append(mse)

                mse_mean = np.mean(mses)
                mse_std = np.std(mses)

                if mse_mean < mse_min:
                    best_model = BestModel(ns, sigma, symmetrical, mse_mean)
                    mse_min = mse_mean

                if verbose:
                    cols = [ns, sigma, symmetrical, mse_mean, mse_std]
                    fmt = "{:<{w}}{:<{w}.2f}{!r:<{w}}{:<{w}.2e}{:<{w}.2e}"

                    print(fmt.format(*cols, w=colwidth), flush=True)

    if verbose:
        print()

    print("Best Model:")
    print("Nodes: {}".format(best_model.nodes))
    print("Std: {:.2f}".format(best_model.node_std))
    print("Symmetrical: {0:}".format(best_model.symmetrical))
    print("MSE: {:1.2e}".format(best_model.mse))

    return best_model


def ballist_evaluate(model,
                     inputs_test,
                     outputs_test,
                     show_points=True,
                     show_surface=True):

    # compute prediction
    outputs_pred = model.predict(inputs_test)
    mse = np.sum((outputs_pred - outputs_test)**2) / outputs_pred.shape[0]

    # set up figure
    fig = figure(1, 1, size=figsize(2, 3))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    cmap = plt.get_cmap('tab10')

    # scatter plots
    x, y = np.hsplit(inputs_test, 2)

    ax1.scatter(x, y, outputs_test,
                color=cmap(0), alpha=0.5,
                label="Ground Truth")

    if show_points:
        ax1.scatter(x, y, outputs_pred,
                    color=cmap(1), alpha=0.5,
                    label="Prediction")

    ax2.scatter(x, y, abs(outputs_pred - outputs_test),
                color=cmap(2), alpha=0.5,
                label="|Prediction - Ground Truth| (MSE is {:1.2e})".format(mse))

    # surface plot
    if show_surface:
        x, y = np.mgrid[0:1.1:0.1, 0:1.1:0.1]

        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        z = np.zeros_like(x)

        for w, mu, cov in zip(model.weights, model.mu, model.cov):
            z += w * multivariate_normal.pdf(pos, mean=mu, cov=cov)

        ax1.plot_wireframe(x, y, z, color=cmap(1), alpha=0.5)

    # format axes
    for ax in ax1, ax2:
        ax.set_xlabel("Angle")
        ax.set_ylabel("Velocity")

        ax.legend()

    ax2.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
