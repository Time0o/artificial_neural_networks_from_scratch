import os

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats


FIGSIZE = (4, 4)
FONTSIZE = 10

_figdir = None


def figsize(rows, cols):
    return FIGSIZE[0] * cols, FIGSIZE[1] * rows


def figure(rows, cols, size=None):
    if size is None:
        size = figsize(rows, cols)

    _prepare_figure()

    return plt.figure(figsize=size)


def subplots(rows, cols, size=None):
    if size is None:
        size = figsize(rows, cols)

    _prepare_figure()

    return plt.subplots(rows, cols, figsize=size)


def set_figure_dir(figdir):
    global _figdir
    _figdir = figdir


def savefig(filename):
    if _figdir is not None:
        plt.savefig(os.path.join(_figdir, filename))


def _prepare_figure():
    set_matplotlib_formats('svg')
    matplotlib.rcParams['font.size'] = FONTSIZE
