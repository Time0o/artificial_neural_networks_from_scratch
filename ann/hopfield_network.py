import sys

import numpy as np
from matplotlib.ticker import MaxNLocator

from .plotting import figsize, subplots
from .patterns import binary_patterns, sparse_patterns

TRAINING_PATTERNS = np.array([
    [-1, -1, 1, -1, 1, -1, -1, 1],
    [-1, -1, -1, -1, -1, 1, -1, -1],
    [-1, 1, 1, -1, -1, 1, -1, 1]
], dtype=np.int8)

DISTORTED_PATTERNS = np.array([
    [1, -1, 1, -1, 1, -1, -1, 1],
    [1, 1, -1, -1, -1, 1, -1, -1],
    [1, 1, 1, -1, 1, 1, -1, 1]
], dtype=np.int8)


class HopfieldNetwork:
    def __init__(self, bias=None):
        self.weights = None
        self.bias = bias
        self._to_update = []

    def train(self,
              patterns,
              synchronous=False,
              activity=None,
              check=True,
              warn=True):

        if activity is not None:
            self.weights = np.sum([
                (p[:, np.newaxis] - activity) @ (p[np.newaxis, :] - activity)
                for p in patterns
            ], axis=0)
        else:
            self.weights = np.sum([
                p[:, np.newaxis] @ p[np.newaxis, :]
                for p in patterns
            ], axis=0)

        self.weights = self.weights.astype(np.float64).copy()

        self.weights /= self.weights.shape[0]

        if check:
            successes = 0
            for pattern in patterns:
                recalled = self.recall(pattern, synchronous=synchronous)

                if not np.array_equal(recalled, pattern):
                    if warn:
                        warn = "warning: input pattern is not a fixed point"
                        print(warn, file=sys.stderr)
                else:
                    successes += 1

            return successes

    def update(self, pattern, synchronous=False):
        if synchronous:
            if self.bias is not None:
                pattern_ = np.sign(self.weights @ pattern - self.bias)
                pattern_[pattern_ == 0] = 1
                pattern_ = 0.5 + 0.5 * pattern_
            else:
                pattern_ = np.sign(self.weights @ pattern)
                pattern_[pattern_ == 0] = 1
        else:
            if not self._to_update:
                self._to_update = [i for i in range(pattern.shape[0])]

            pattern_ = pattern.copy()

            j = np.random.choice(self._to_update)

            if self.bias is not None:
                upd = np.sign(np.sum(self.weights[j, :] * pattern) - self.bias)
                pattern_[j] = 0.5 + 0.5 * (1 if upd >= 0 else -1)
            else:
                upd = np.sign(np.sum(self.weights[j, :] * pattern))
                pattern_[j] = 1 if upd >= 0 else -1

            self._to_update.remove(j)

        return pattern_

    def recall(self,
               pattern,
               synchronous=False,
               iterations=None,
               return_iterations=False):

        if self.weights is None:
            raise ValueError("network not trained yet")

        i = 0
        while True:
            if synchronous:
                pattern_ = self.update(pattern, synchronous=True)
            else:
                pattern_ = pattern
                for _ in range(pattern.shape[0]):
                    pattern_ = self.update(pattern_, synchronous=False)

            if np.array_equal(pattern_, pattern):
                break

            i += 1
            if iterations is not None and i == iterations:
                break

            pattern = pattern_

        if return_iterations:
            return pattern, i
        else:
            return pattern

    def attractors(self, synchronous=False):
        if self.weights is None:
            raise ValueError("network not trained yet")

        attractors = set()

        for pattern in binary_patterns(len(self.weights)):
            attractors.add(tuple(self.recall(pattern, synchronous=synchronous)))

        return np.array([np.array(attr, dtype=np.int8) for attr in attractors])

    def energy(self, state):
        if self.weights is None:
            raise ValueError("network not trained yet")

        alpha = state[:, np.newaxis] @ state[np.newaxis, :]

        return -np.sum(alpha * self.weights)


def profile_noise_resistance(model,
                             patterns,
                             noise_levels=None,
                             synchronous=True,
                             plot_iterations=True,
                             plot_fixpoints=True,
                             axes=None,
                             silent=False):

    if noise_levels is None:
        noise_levels = range(101)

    # initialize plots
    if plot_iterations and plot_fixpoints:
        if axes is None:
            _, axes = subplots(2 + len(patterns), 1,
                               size=figsize(2 + len(patterns), 3))
        ax_rt = axes[0]
        ax_iter = axes[1]
        axes_attr = axes[2:]

    elif plot_iterations:
        if axes is None:
            _, (ax_rt, ax_iter) = subplots(2, 1, size=figsize(2, 3))
        else:
            ax_rt, ax_iter = axes

    elif plot_fixpoints:
        if axes is None:
            _, axes = subplots(1 + len(patterns), 1,
                               size=figsize(1 + len(patterns), 3))
        ax_rt = axes[0]
        axes_attr = axes[1:]

    else:
        if axes is None:
            _, ax_rt = subplots(1, 1, size=figsize(1, 3))
        else:
            ax_rt = axes

    # start profiling
    attractors = patterns.copy()

    for i, p in enumerate(patterns):
        p = p.copy()

        recovery_rate = []

        if plot_iterations:
            iterations = []

        if plot_fixpoints:
            fixpoints = []

        for noise in noise_levels:
            faults = int(noise / 100 * np.prod(p.shape))

            corrected = 0

            if plot_iterations:
                iterations_ = []

            if plot_fixpoints:
                fixpoints_ = []

            for _ in range(100):
                # create noisy version of attractor
                p_noisy = p.astype(np.int8).copy()

                fault_ind = np.random.choice(len(p), size=faults, replace=False)
                p_noisy[fault_ind] *= -1

                # try to recall noisy attractor
                p_recalled, j = model.recall(
                    p_noisy, synchronous=synchronous, return_iterations=True)

                # record which attractor the algorithm has converged to
                if plot_fixpoints:
                    new_fixpoint = True
                    for k in range(len(attractors)):
                        if np.array_equal(p_recalled, attractors[k]):
                            fixpoints_.append(k)
                            new_fixpoint = False
                            break

                    if new_fixpoint:
                        attractors.append(p_recalled)
                        fixpoints_.append(len(attractors) - 1)

                if np.array_equal(p_recalled, p):
                    corrected += 1

                    if plot_iterations:
                        iterations_.append(j)

            recovery_rate.append(corrected)

            if plot_iterations:
                iterations.append(iterations_)

            if plot_fixpoints:
                fixpoints.append(fixpoints_)

            if not silent:
                sys.stdout.write("Attractor {}: {}\r".format(i + 1, noise))
                sys.stdout.flush()

        label = "Attractor {}".format(i + 1)

        # plot recovery rates
        ax_rt.plot(noise_levels, recovery_rate, label=label)

        # plot recovery iterations
        if plot_iterations:
            i_mean = [np.mean(i) if i else np.nan for i in iterations]
            i_std = [np.std(i) if i else np.nan for i in iterations]
            ax_iter.errorbar(noise_levels, i_mean, yerr=i_std, label=label)

        # visualize attractors reached
        if plot_fixpoints:
            labels = ["Attractor {}".format(i + 1)
                      for i in range(len(attractors))]

            stacks = []
            for a in range(len(attractors)):
                stacks.append([np.count_nonzero(np.array(fp) == a)
                               for fp in fixpoints])

            axes_attr[i].stackplot(noise_levels, stacks, labels=labels)

        if not silent:
            print()

    ax_rt.set_xlabel("Distorted Pixels (Percent)")
    ax_rt.set_ylabel("Successful Restorations (Percent)")
    ax_rt.legend()
    ax_rt.grid()

    if plot_iterations:
        ax_iter.set_xlabel("Distorted Pixels (Percent)")
        ax_iter.set_ylabel("Iterations to Successful Restoration")
        ax_iter.legend()
        ax_iter.grid()

    if plot_fixpoints:
        for i, ax in enumerate(axes_attr):
            ax.set_title("(Distorted) Attractor {}".format(i + 1))
            ax.set_xlabel("Distorted Pixels (Percent)")
            ax.set_ylabel("Attractors Reached (Percent)")

            bbox = ax.get_position()
            ax.set_position([bbox.x0, bbox.y0, bbox.width * 0.7, bbox.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            ax.margins(0, 0)


def profile_sparse_storage(num_patterns_var,
                           num_patterns_fixed,
                           biases_var,
                           biases_fixed,
                           pattern_size=300,
                           pattern_activity=0.1,
                           synchronous=True,
                           axes=None,
                           silent=False):

    if axes is None:
        _, axes = subplots(1, 2, size=figsize(2, 3))

    for bias in biases_fixed:
        model = HopfieldNetwork(bias=bias)

        success_rates = []
        for n in num_patterns_var:
            if not silent:
                sys.stdout.write("{} patterns, bias = {:.2f}\r".format(n, bias))
                sys.stdout.flush()

            successes = 0
            for _ in range(100):
                patterns = sparse_patterns(
                    size=pattern_size, n=n, activity=pattern_activity)

                res = model.train(patterns,
                                  synchronous=synchronous,
                                  activity=pattern_activity,
                                  warn=False)

                successes += res == len(patterns)

            if successes == 0:
                break

            success_rates.append(successes)

        if len(success_rates) < len(num_patterns_var):
            success_rates += [0] * (len(num_patterns_var) - len(success_rates))

        axes[0].plot(num_patterns_var, success_rates,
                     label=r"$\Theta = {:.3f}$".format(bias))

    if not silent:
        print()

    for n in num_patterns_fixed:
        success_rates = []

        for bias in biases_var:
            if not silent:
                sys.stdout.write("{} patterns, bias = {:.2f}\r".format(n, bias))
                sys.stdout.flush()

            model = HopfieldNetwork(bias=bias)

            successes = 0
            for _ in range(100):
                patterns = sparse_patterns(
                    size=pattern_size, n=n, activity=pattern_activity)

                res = model.train(patterns,
                                  synchronous=synchronous,
                                  activity=pattern_activity,
                                  warn=False)

                successes += res == len(patterns)

            success_rates.append(successes)

        axes[1].plot(biases_var, success_rates,
                     label="{} Patterns".format(n))

    for ax in axes:
        ax.set_title("{} Updates".format(
            "Synchronous" if synchronous else "Asynchronous"))

    axes[0].set_xlabel("Number of Patterns")
    axes[0].set_ylabel("Success Rate (Percent)")

    axes[1].set_xlabel("$\Theta$")
    axes[1].set_ylabel("Success Rate (Percent)")

    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    for ax in axes:
        ax.legend()
        ax.grid()
