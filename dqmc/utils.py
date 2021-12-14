# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import time as _time
import matplotlib.pyplot as plt


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
    def pause(interval=0.0):
        plt.pause(interval)

    def draw(self, pause=1e-10):
        self.fig.canvas.flush_events()
        plt.show(block=False)
        plt.pause(pause)


class Timer:

    def __init__(self):
        self.t0 = 0

    def start(self):
        self.t0 = _time.perf_counter()

    def time(self):
        if not self.t0:
            raise RuntimeError("Timer has not been started yet!")
        return _time.perf_counter() - self.t0

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = self.time()
        s = f"Total time: {t:.2f}s"
        line = "-" * (len(s) + 1)
        print(line)
        print(s)
