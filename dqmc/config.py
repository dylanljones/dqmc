# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

UP, DN = +1, -1


def init_configuration(num_sites: int, num_timesteps: int) -> np.ndarray:
    """Initializes the configuration array with a random distribution of `-1` and `+1`.

    Parameters
    ----------
    num_sites : int
        The number of sites `N` of the lattice model.
    num_timesteps : int
        The number of time steps `L` used in the Monte Carlo simulation.

    Returns
    -------
    config : (N, L) np.ndarray
        The array representing the configuration or or Hubbard-Stratonovich field.
    """
    return 2 * np.random.randint(0, 2, size=(num_sites, num_timesteps), dtype=np.int8) - 1


@njit
def update_configuration(config: np.ndarray, i: int, t: int) -> None:
    """Updates an element of the HS-field by flipping it's spin-value.

    Parameters
    ----------
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    i : int
        The index of the lattice site to update.
    t : int, optional
        The index of the time slice to update.
    """
    config[i, t] = -config[i, t]


class ConfigurationPlot:

    def __init__(self, config, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            ax.invert_yaxis()
        else:
            fig = ax.get_figure()
        self.fig = fig
        self.ax = ax
        self.im = None

        self.plot_config(config)

    def set_figsize(self, width, height, dpi=None):
        self.fig.set_size_inches(width, height)
        if dpi is not None:
            self.fig.set_dpi(dpi)

    def tight_layout(self):
        self.fig.tight_layout()

    def plot_config(self, config):
        self.im = self.ax.imshow(config)

    def update_config(self, config):
        self.im.set_data(config)

    def show(self, tight=True, block=True):
        if tight:
            self.tight_layout()
        plt.show(block=block)

    @staticmethod
    def pause(interval=0.):
        plt.pause(interval)

    def draw(self, pause=1e-10):
        self.fig.canvas.flush_events()
        plt.show(block=False)
        plt.pause(pause)
