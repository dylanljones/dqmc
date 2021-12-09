# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

# flake8: noqa

import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dqmc import DQMC, hubbard_hypercube, mfuncs

logger = logging.getLogger("dqmc")
logger.setLevel(logging.WARNING)


def build_temperature_array(temps, model, num_timesteps, warmup=500, measure=3000,
                            callback=None):
    out = np.zeros((len(temps), model.num_sites))
    for i in tqdm(range(len(temps)), leave=False):
        model.set_temperature(temps[i])
        dqmc = DQMC(model, num_timesteps)
        mom = dqmc.simulate(warmup, measure, callback=callback)
        out[i] = mom
    return out


def build_inter_temp_array(inters, temps, model, num_timesteps, warmup=500,
                           measure=3000, callback=None):
    out = np.zeros((len(inters), len(temps), model.num_sites))
    for i in tqdm(range(len(inters))):
        model.u = inters[i]
        out[i] = build_temperature_array(temps, model, num_timesteps,
                                         warmup, measure, callback)
    return out


def main():
    shape = 10
    u, mu, hop = 4.0, 2.0, 1.0
    inters = [4, 5, 6, 7, 8]  # np.arange(5, 10, 1.0)
    temps = np.geomspace(0.2, 100, 15)
    num_timesteps = 50
    warmup, measure = 500, 3000
    model = hubbard_hypercube(shape, u=u, mu=mu, hop=hop, beta=1.0, periodic=True)

    # moments = build_inter_temp_array(inters, temps, model, num_timesteps, warmup,
    #                                  measure, callback=mfuncs.mz_moment)
    # np.savez("mz2_chain.npz", moments=moments)
    moments = np.load("mz2_chain.npz")["moments"]

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    for u, mom in zip(inters[::-1], moments[::-1]):
        ax.plot(temps, np.mean(mom.T, axis=0), marker="o", label=f"$U=${u}")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\langle m_z^2 \rangle$")
    ax.grid()
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
