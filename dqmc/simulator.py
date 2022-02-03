# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Main DQMC simulator object, see `dqmc` for implementation of the DQMC methods."""

import time
import logging
import numpy as np
from tqdm import tqdm
from .model import hubbard_hypercube
from .dqmc import (     # noqa: F401
    init_qmc,
    compute_timestep_mats,
    compute_greens,
    compute_greens_qrd,
    compute_unequal_time_greens,
    init_greens,
    dqmc_iteration,
)
from .measurements import MeasurementData


logger = logging.getLogger("dqmc")


class DQMC:
    """Main DQMC simulator instance."""

    def __init__(self, model, num_times, num_recomp=1, prod_len=1, seed=None,
                 sampl_recomp=True, progress=False, unequal_time=True):
        if num_recomp > 0 and num_times % num_recomp != 0:
            raise ValueError("Number of time steps not a multiple of `num_recomp`!")
        if prod_len > 0 and num_times % prod_len != 0:
            raise ValueError("Number of time steps not a multiple of `prod_len`!")

        if seed is None:
            seed = 0
        # random.seed(seed)
        self.progress = progress
        self.unequal_time_measurements = unequal_time

        self.num_recomp = num_recomp
        self.sampl_recomp = sampl_recomp
        self.prod_len = prod_len

        self.model = model
        # Init QMC variables
        self.expk, self.expk_inv, self.nu, self.config = init_qmc(model, num_times,
                                                                  seed)

        # Pre-compute time flow matrices
        self.bmats_up, self.bmats_dn = compute_timestep_mats(
            self.expk, self.nu, self.config
        )

        # Initialize QMC statistics
        self.it = 0
        self.status = ""
        self.acceptance_probs = list()

        # Initialization
        gf_up, gf_dn, sgndet, logdet = init_greens(self.bmats_up, self.bmats_dn,
                                                   0, self.prod_len)
        self.gf_up = gf_up
        self.gf_dn = gf_dn
        self.sgndet = sgndet
        self.logdet = logdet

        # Measurement data
        num_sites = self.config.shape[0]
        self.measurements = MeasurementData(num_sites, num_times)

    def compute_greens(self, t=0):   # noqa: F811
        if self.prod_len == 0:
            compute_greens(
                self.bmats_up,
                self.bmats_dn,
                self.gf_up,
                self.gf_dn,
                self.sgndet,
                self.logdet,
                t
            )
        else:
            compute_greens_qrd(
                self.bmats_up,
                self.bmats_dn,
                self.gf_up,
                self.gf_dn,
                self.sgndet,
                self.logdet,
                t,
                self.prod_len
            )

    def get_greens(self):
        return self.gf_up, self.gf_dn

    def iteration(self):
        accepted = dqmc_iteration(
            self.expk,
            self.nu,
            self.config,
            self.bmats_up,
            self.bmats_dn,
            self.gf_up,
            self.gf_dn,
            self.sgndet,
            self.logdet,
            self.num_recomp,
            self.prod_len
        )
        # Compute and save acceptance ratio
        acc_ratio = accepted / self.config.size
        self.acceptance_probs.append(acc_ratio)
        logger.debug("[%s] %3d Ratio: %.2f  Signs: (%+d %+d)",
                     self.status, self.it, acc_ratio, self.sgndet[0], self.sgndet[1])

    def accumulate_measurements(self):
        if self.sampl_recomp:
            # Recompute Green's functions before measurements
            self.compute_greens()
        self.measurements.accumulate(self.gf_up, self.gf_dn, self.sgndet)
        if self.unequal_time_measurements:
            self.measurements.accumulate_unequal_time(self.bmats_up,
                                                      self.bmats_dn,
                                                      self.gf_up,
                                                      self.gf_dn,
                                                      self.sgndet)

    def warmup(self, sweeps):
        self.it = 0
        self.status = "warm"
        for _ in tqdm(range(sweeps), desc="Warmup", disable=not self.progress):
            self.iteration()
            self.it += 1

    def measure(self, sweeps, callback=None, *args, **kwargs):
        out = 0.0
        self.status = "meas"
        for _ in tqdm(range(sweeps), desc="Sample", disable=not self.progress):
            self.iteration()
            # perform measurements
            self.accumulate_measurements()
            # user measurement callback
            if callback is not None:
                out += np.asarray(callback(self, *args, **kwargs))
            self.it += 1
        return out / sweeps

    def simulate(self, num_equil, num_sampl, callback=None, *args, **kwargs):
        total_sweeps = num_equil + num_sampl
        t0 = time.perf_counter()

        logger.info("Running %s equilibrium sweeps...", num_equil)
        t0_equil = time.perf_counter()
        self.warmup(num_equil)
        t_equil = time.perf_counter() - t0_equil

        logger.info("Running %s sampling sweeps...", num_sampl)
        t0_sampl = time.perf_counter()
        extra_results = self.measure(num_sampl, callback, *args, **kwargs)
        t_sampl = time.perf_counter() - t0_sampl

        t = time.perf_counter() - t0
        logger.info("%s iterations completed!", total_sweeps)
        logger.info("    Signs: [%+d  %+d]", self.sgndet[0], self.sgndet[1])
        logger.info(" Log Dets: [%.2f  %.2f]", self.logdet[0], self.logdet[1])
        logger.info("Equil CPU time: %6.1fs  (%.4f s/it)", t_equil, t_equil / num_equil)
        logger.info("Sampl CPU time: %6.1fs  (%.4f s/it)", t_sampl, t_sampl / num_sampl)
        logger.info("Total CPU time: %6.1fs  (%.4f s/it)", t, t / total_sweeps)
        return self.measurements, extra_results


def init_simulator(p, progress=False):
    model = hubbard_hypercube(p.shape, p.u, p.eps, p.t, p.mu, p.beta, periodic=True)
    return DQMC(model, p.num_times, p.num_wraps, p.prod_len, p.seed,
                bool(p.sampl_recomp), progress)


def run_dqmc(p, unequal_time=True, callback=None, progress=False, *args, **kwargs):
    """Runs a DQMC simulation.

    Parameters
    ----------
    p : Parameters
        The input parameters of the DQMC simulation.
    unequal_time : bool
        If `True`, unequal-time measurements will be performed.
    callback : callable, optional
        A optional callback method for measuring additional observables.
    progress : bool
        If `True` a progressbar will be printed.
    *args : tuple, optional
        Optional positional arguments for the user callback method.
    **kwargs : dict, optional
        Optional keyword arguments for the user callback method.

    Returns
    -------
    results : Tuple
        The results of the DQMC simulation. The last item is the result of the user
        callback or `None`.
    """
    model = hubbard_hypercube(p.shape, p.u, p.eps, p.t, p.mu, p.beta, periodic=True)
    dqmc = DQMC(model, p.num_times, p.num_wraps, p.prod_len, p.seed,
                bool(p.sampl_recomp), progress, unequal_time)
    try:
        results, extra = dqmc.simulate(p.num_equil, p.num_sampl, callback=callback,
                                       *args, **kwargs)
    except np.linalg.LinAlgError:
        return ()
    # results = [dqmc.gf_up, dqmc.gf_dn, dqmc.n_up, dqmc.n_dn, dqmc.n_double,
    #            dqmc.local_moment, extra_results]
    # return results
    result_arr = list(results.normalize()) + [extra]
    return result_arr


def log_results(*results):
    n_up = np.mean(results[2])
    n_dn = np.mean(results[3])
    n_double = np.mean(results[4])
    local_moment = np.mean(results[5])

    logger.info("_" * 60)
    logger.info("Simulation results")
    logger.info("")
    logger.info("     Total density: %8.4f", n_up + n_dn)
    logger.info("   Spin-up density: %8.4f", n_up)
    logger.info(" Spin-down density: %8.4f", n_dn)
    logger.info("  Double occupancy: %8.4f", n_double)
    logger.info("      Local moment: %8.4f", local_moment)
    logger.info("")
