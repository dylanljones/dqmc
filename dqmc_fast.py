# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dqmc import DQMC, hubbard_hypercube, mfuncs

logger = logging.getLogger("dqmc")
logger.setLevel(logging.WARNING)


def build_temperature_array(temps, model, num_timesteps, warmup=500, measure=3000, callback=None):
    out = np.zeros((len(temps), model.num_sites))
    for i in tqdm(range(len(temps))):
        model.set_temperature(temps[i])
        dqmc = DQMC(model, num_timesteps)
        mom = dqmc.simulate(warmup, measure, callback=callback)
        out[i] = mom
    return out


def main():
    shape = (4, 4)
    u, mu, hop = 4.0, 0.0, 1.0
    temps = np.geomspace(0.2, 100, 50)
    num_timesteps = 50
    warmup, measure = 500, 2000
    model = hubbard_hypercube(shape, u=u, mu=mu, hop=hop, beta=1.0, periodic=(0,1))

    moments = build_temperature_array(temps, model, num_timesteps, warmup, measure,
                                      callback=mfuncs.mz_moment)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.plot(temps, np.mean(moments.T, axis=0))
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\langle m_z^2 \rangle$")
    plt.show()


if __name__ == "__main__":
    main()
