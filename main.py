# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import logging
import numpy as np
import matplotlib.pyplot as plt
from dqmc import DQMC, hubbard_hypercube, mfuncs

logger = logging.getLogger("dqmc")
logger.setLevel(logging.INFO)


def plot_acceptance_probs(probs, warmup):
    fig, ax = plt.subplots()
    ax.plot(probs)
    ax.grid()
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlim(0, None)
    ax.set_xlabel("Seeep")
    ax.set_ylabel("Ratio")
    ax.axvline(x=warmup, color="r", ls="--")


def run_dqmc(num_sites, num_timesteps, u, temp, mu=0., hop=1., warmup=500, measure=2000,
             callback=None):
    model = hubbard_hypercube(num_sites, u=u, mu=mu, hop=hop, beta=1 / temp, periodic=True)
    dqmc = DQMC(model, num_timesteps)
    return dqmc.simulate(warmup, measure, callback), dqmc.acceptance_probs


def pformat_result(value, error, dec=5):
    return f"{value:.{dec}f}±{error:.{dec}f}"


def main():
    num_sites = 10
    num_timesteps = 100
    warmup = 500
    measure = 3000
    u, mu, hop = 4.0, 0.0, 1.0
    temp = 10

    res, probs = run_dqmc(num_sites, num_timesteps, u, temp, mu, hop, warmup, measure, mfuncs.mz_moment)
    print(res)
    print(pformat_result(np.mean(res), np.std(res), dec=4))
    plot_acceptance_probs(probs, warmup)
    plt.show()


if __name__ == "__main__":
    main()
