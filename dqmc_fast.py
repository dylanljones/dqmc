# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import logging
import numpy as np
import matplotlib.pyplot as plt
from dqmc.dqmc import init_qmc, compute_greens, compute_timestep_mats, iteration_fast
from dqmc.model import hubbard_hypercube
from dqmc import mfuncs
from tqdm import tqdm

logger = logging.getLogger("dqmc")
logger.setLevel(logging.WARNING)


class DQMC:

    def __init__(self, model, num_timesteps, time_dir=+1, bmat_dir=None):
        # Init QMC variables
        self.exp_k, self.nu, self.config = init_qmc(model, num_timesteps)

        # Set up time direction and order of inner loops
        if bmat_dir is None:
            # Default B-matrix direction is the reverse of the time direction
            bmat_dir = - time_dir
        self.bmat_order = np.arange(self.config.shape[1], dtype=np.int64)[::bmat_dir]
        self.times = np.arange(self.config.shape[1], dtype=np.int64)[::time_dir]
        self.sites = np.arange(self.config.shape[0], dtype=np.int64)

        # Pre-compute time flow matrices
        self.bmats_up, self.bmats_dn = compute_timestep_mats(self.exp_k, self.nu, self.config)

        # Initialize QMC statistics
        self.status = ""
        self.acceptance_probs = list()

        # Initialization
        self._gf_up, self._gf_dn = self.greens()

    def recompute_greens(self):
        self._gf_up, self._gf_dn = self.greens()

    def greens(self):
        return compute_greens(self.bmats_up, self.bmats_dn, self.bmat_order)

    def get_greens(self):
        return self._gf_up, self._gf_dn

    def iteration(self):
        accepted = iteration_fast(self.exp_k, self.nu, self.config, self.bmats_up, self.bmats_dn,
                                  self._gf_up, self._gf_dn, self.times)
        # Compute and save acceptance ratio
        acc_ratio = accepted / self.config.size
        self.acceptance_probs.append(acc_ratio)
        # Recompute Green's functions
        self.recompute_greens()

    def simulate(self, warmup, measure, callback):
        sweeps = warmup + measure
        out = 0.
        # Run sweeps
        self.status = "warmup"
        for sweep in range(sweeps):
            self.iteration()
            logger.info("[%s] %3d Ratio: %.2f", self.status, sweep, self.acceptance_probs[-1])
            # perform measurements
            if sweep > warmup:
                self.status = "measurements"
                gf_up, gf_dn = self.get_greens()
                if callback is not None:
                    out += callback(gf_up, gf_dn)
                else:
                    out += np.array([gf_up, gf_dn])
        return out / measure


def build_temperature_array(temps, model, num_timesteps, warmup=500, measure=3000, callback=None):
    out = np.zeros((len(temps), model.num_sites))
    for i in tqdm(range(len(temps))):
        model.set_temperature(temps[i])
        dqmc = DQMC(model, num_timesteps)
        mom = dqmc.simulate(warmup, measure, callback=callback)
        out[i] = mom
    return out


def main():
    shape = 4, 4
    u, mu, hop = 4.0, 0.0, 1.0
    temps = np.geomspace(0.2, 100, 50)
    num_timesteps = 50
    warmup, measure = 500, 2000
    model = hubbard_hypercube(shape, u=u, mu=mu, hop=hop, beta=1.0, periodic=(0, 1))

    moments = build_temperature_array(temps, model, num_timesteps, warmup, measure,
                                      callback=mfuncs.mz_moment)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.plot(temps, np.sum(moments.T, axis=0))
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\langle m_z^2 \rangle$")
    plt.show()


if __name__ == "__main__":
    main()
