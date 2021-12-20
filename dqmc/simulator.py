# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Main DQMC simulator object, see `dqmc` for implementation of the DQMC methods."""

import numpy as np
import logging
from .model import hubbard_hypercube
from .dqmc import init_qmc, compute_timestep_mats, compute_greens, iteration_fast
from .stabilize import compute_greens_stable
from .mp import run_parallel, prun_parallel

logger = logging.getLogger("dqmc")


class DQMC:
    """Main DQMC simulator instance."""

    def __init__(self, model, num_timesteps, time_dir=+1, bmat_dir=None, seed=None):
        if seed is None:
            seed = 0
        # random.seed(seed)
        # np.random.seed(seed)

        self.model = model
        # Init QMC variables
        self.exp_k, self.nu, self.config = init_qmc(model, num_timesteps)

        # Set up time direction and order of inner loops
        if bmat_dir is None:
            # Default B-matrix direction is the reverse of the time direction
            bmat_dir = -time_dir

        self.bmat_order = np.arange(self.config.shape[1], dtype=np.int64)[::bmat_dir]
        # self.bmat_order = np.roll(self.bmat_order, 1)
        self.times = np.arange(self.config.shape[1], dtype=np.int64)[::time_dir]

        # Pre-compute time flow matrices
        self.bmats_up, self.bmats_dn = compute_timestep_mats(
            self.exp_k, self.nu, self.config
        )

        # Initialize QMC statistics
        self.it = 0
        self.status = ""
        self.acceptance_probs = list()

        # Initialization
        self._gf_up, self._gf_dn = self.greens()

    def recompute_greens(self):
        self._gf_up, self._gf_dn = self.greens()

    def greens(self):
        return compute_greens_stable(self.bmats_up, self.bmats_dn, self.bmat_order)
        try:
            return compute_greens(self.bmats_up, self.bmats_dn, self.bmat_order)
        except np.linalg.LinAlgError as e:
            logger.warning("Error in iteration %s", self.it)
            logger.warning(e)
            return compute_greens_stable(self.bmats_up, self.bmats_dn, self.bmat_order)

    def get_greens(self):
        return self._gf_up, self._gf_dn

    def iteration(self):
        accepted = iteration_fast(
            self.exp_k,
            self.nu,
            self.config,
            self.bmats_up,
            self.bmats_dn,
            self._gf_up,
            self._gf_dn,
            self.times,
        )
        # Compute and save acceptance ratio
        acc_ratio = accepted / self.config.size
        self.acceptance_probs.append(acc_ratio)
        # Recompute Green's functions
        self.recompute_greens()

    def simulate(self, warmup, measure, callback=None, *args, **kwargs):
        sweeps = warmup + measure
        out = 0.0
        # Run sweeps
        self.it = 0
        self.status = "warm"
        for sweep in range(sweeps):
            # print("It")
            self.iteration()
            logger.debug(
                "[%s] %3d Ratio: %.2f", self.status, sweep, self.acceptance_probs[-1]
            )
            # perform measurements
            if sweep > warmup:
                self.status = "meas"
                gf_up, gf_dn = self.get_greens()
                if callback is not None:
                    out += callback(gf_up, gf_dn, *args, **kwargs)
                else:
                    out += np.array([gf_up, gf_dn])
            self.it += 1
        return out / measure


def run_dqmc(shape, u, eps, hop, mu, beta, num_timesteps, warmup, measure, callback):
    model = hubbard_hypercube(shape, u, eps, hop, mu, beta, periodic=True)
    dqmc = DQMC(model, num_timesteps)
    try:
        return dqmc.simulate(warmup, measure, callback=callback)
    except np.linalg.LinAlgError:
        return np.nan


def run_dqmc_parallel(params, max_workers=None):
    return run_parallel(run_dqmc, params, max_workers)


def prun_dqmc_parallel(params, max_workers=None):
    return prun_parallel(run_dqmc, params, max_workers)
