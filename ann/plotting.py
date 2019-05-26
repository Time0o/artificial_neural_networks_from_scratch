import os

import matplotlib.pyplot as plt


FIGSIZE = (6, 6)

_figdir = None


def figsize(rows, cols):
    return FIGSIZE[0] * cols, FIGSIZE[1] * rows


def figure(rows, cols, size=None):
    if size is None:
        size = figsize(rows, cols)

    return plt.figure(figsize=size)


def subplots(rows, cols, size=None):
    if size is None:
        size = figsize(rows, cols)

    return plt.subplots(rows, cols, figsize=size)


def set_figure_dir(figdir):
    global _figdir
    _figdir = figdir


def savefig(filename):
    if _figdir is not None:
        plt.savefig(os.path.join(_figdir, filename))
