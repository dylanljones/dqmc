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

import random
import logging
import numpy as np
from .model import hubbard_hypercube
from .dqmc import init_qmc, compute_timestep_mats, compute_greens, iteration_fast
from .stabilize import compute_greens_stable
from .mp import run_parallel, prun_parallel

logger = logging.getLogger("dqmc")


class DQMC:
    """Main DQMC simulator instance."""

    def __init__(self, model, num_timesteps, num_recomp=1, seed=None):
        if seed is None:
            seed = 0
        random.seed(seed)

        self.num_recomp = num_recomp

        self.model = model
        # Init QMC variables
        self.exp_k, self.nu, self.config = init_qmc(model, num_timesteps)

        self.bmat_order = np.arange(self.config.shape[1], dtype=np.int64)[::-1]

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

        # Measurement data
        # ----------------

        self.n_up = 0
        self.n_dn = 0
        self.n_double = 0
        self.local_moment = 0

    def recompute_greens(self):
        self._gf_up, self._gf_dn = self.greens()

    def greens(self):
        # return compute_greens_stable(self.bmats_up, self.bmats_dn, self.bmat_order)
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
        )
        # Compute and save acceptance ratio
        acc_ratio = accepted / self.config.size
        self.acceptance_probs.append(acc_ratio)
        logger.debug("[%s] %3d Ratio: %.2f", self.status, self.it, acc_ratio)

        # Recompute Green's functions
        if self.it % self.num_recomp == 0:
            logger.debug("Recomputing GF")
            self.recompute_greens()

    def accumulate_measurements(self, factor):
        scale = 1 / factor

        gf_up, gf_dn = self.get_greens()
        n_up = 1 - np.diag(gf_up)
        n_dn = 1 - np.diag(gf_dn)
        n_double = n_up * n_dn
        moment = n_up + n_dn - 2 * n_double

        self.n_up += np.mean(n_up) * scale
        self.n_dn += np.mean(n_dn) * scale
        self.n_double += np.mean(n_double) * scale
        self.local_moment += np.mean(moment) * scale

    def warmup(self, sweeps):
        self.it = 0
        self.status = "warm"
        for sweep in range(sweeps):
            self.iteration()
            self.it += 1

    def measure(self, sweeps, callback=None, *args, **kwargs):
        out = 0.0
        self.status = "meas"
        for sweep in range(sweeps):
            self.iteration()

            # perform measurements
            self.accumulate_measurements(sweeps)
            # user measurement callback
            if callback is not None:
                gf_up, gf_dn = self.get_greens()
                out += callback(gf_up, gf_dn, *args, **kwargs)

            self.it += 1

    def simulate(self, warmup, measure, callback=None, *args, **kwargs):
        if (warmup + measure) % self.num_recomp != 0:
            raise ValueError("Number of sweeps not a multiple of `num_recomp`!")
        self.warmup(warmup)
        return self.measure(measure, callback, *args, **kwargs)


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
