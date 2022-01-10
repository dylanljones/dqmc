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

import time
import logging
import numpy as np
from typing import Union
from dataclasses import dataclass
from .model import hubbard_hypercube
from .dqmc import (     # noqa: F401
    init_qmc,
    compute_timestep_mats,
    compute_greens,
    compute_greens_qrd,
    init_greens,
    dqmc_iteration,
    accumulate_measurements,
)

logger = logging.getLogger("dqmc")


@dataclass
class Parameters:

    shape: Union[int, tuple]
    u: float
    eps: float
    t: float
    mu: float
    dt: float
    num_timesteps: int
    num_equil: int = 512
    num_sampl: int = 2048
    num_wraps: int = 1
    sampl_recomp: int = 1
    prod_len: int = 1
    seed: int = 0

    def copy(self, **kwargs):
        # Copy parameters
        p = Parameters(**self.__dict__)
        # Update new parameters with given kwargs
        for key, val in kwargs.items():
            setattr(p, key, val)
        return p

    @property
    def beta(self):
        return self.num_timesteps * self.dt

    @beta.setter
    def beta(self, beta):
        self.dt = beta / self.num_timesteps

    @property
    def temp(self):
        return 1 / (self.num_timesteps * self.dt)

    @temp.setter
    def temp(self, temp):
        self.dt = 1 / (temp * self.num_timesteps)


def parse(file):
    shape = 0
    u = 0.
    eps = 0.
    t = 0.
    mu = 0.
    dt = 0.
    beta = 0.
    temp = 0.
    num_timesteps = 0
    warm = 0
    meas = 0
    num_recomp = 0
    sampl_recomp = 1
    prod_len = 1

    logger.info("Reading file %s...", file)
    with open(file, "r") as fh:
        text = fh.read()
    lines = text.splitlines(keepends=False)
    for line in lines:
        if "#" in line:
            text, comm = line.strip().split("#")
            line = text.strip()
        if not line:
            continue

        head, val = line.split(maxsplit=1)
        head = head.lower()
        if head == "shape":
            shape = tuple(int(x) for x in val.split(", "))
        elif head == "u":
            u = float(val)
        elif head == "eps":
            eps = float(val)
        elif head == "t":
            t = float(val)
        elif head == "mu":
            mu = float(val)
        elif head == "dt":
            dt = float(val)
        elif head == "l":
            num_timesteps = int(val)
        elif head == "nequil":
            warm = int(val)
        elif head == "nsampl":
            meas = int(val)
        elif head == "nwraps":
            num_recomp = int(val)
        elif head == "recomp":
            sampl_recomp = int(val)
        elif head == "prodlen":
            prod_len = int(val)
        elif head == "beta":
            beta = float(val)
        elif head == "temp":
            temp = float(val)
        else:
            logger.warning("Parameter %s of file '%s' not recognized!", head, file)
    if temp:
        beta = 1 / temp
    if dt == 0:
        dt = beta / num_timesteps
    elif num_timesteps == 0:
        num_timesteps = int(beta / dt)
    return Parameters(shape, u, eps, t, mu, dt, num_timesteps, warm, meas,
                      num_recomp, sampl_recomp, prod_len)


class DQMC:
    """Main DQMC simulator instance."""

    def __init__(self, model, num_timesteps, num_recomp=1, prod_len=1, seed=None,
                 sampl_recomp=True):
        if num_recomp > 0 and num_timesteps % num_recomp != 0:
            raise ValueError("Number of time steps not a multiple of `num_recomp`!")
        if prod_len > 0 and num_timesteps % prod_len != 0:
            raise ValueError("Number of time steps not a multiple of `prod_len`!")

        if seed is None:
            seed = 0
        # random.seed(seed)

        self.num_recomp = num_recomp
        self.sampl_recomp = sampl_recomp
        self.prod_len = prod_len

        self.model = model
        # Init QMC variables
        self.expk, self.expk_inv, self.nu, self.config = init_qmc(model, num_timesteps,
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
        self._gf_up = gf_up
        self._gf_dn = gf_dn
        self._sgndet = sgndet
        self._logdet = logdet

        # Measurement data
        # ----------------
        num_sites = self.config.shape[0]
        self.n_up = np.zeros(num_sites, dtype=np.float64)
        self.n_dn = np.zeros(num_sites, dtype=np.float64)
        self.n_double = np.zeros(num_sites, dtype=np.float64)
        self.local_moment = np.zeros(num_sites, dtype=np.float64)

    def compute_greens(self, t=0):   # noqa: F811
        if self.prod_len == 0:
            compute_greens(
                self.bmats_up,
                self.bmats_dn,
                self._gf_up,
                self._gf_dn,
                self._sgndet,
                self._logdet,
                t
            )
        else:
            compute_greens_qrd(
                self.bmats_up,
                self.bmats_dn,
                self._gf_up,
                self._gf_dn,
                self._sgndet,
                self._logdet,
                t,
                self.prod_len
            )

    def get_greens(self):
        return self._gf_up, self._gf_dn

    def iteration(self):
        accepted = dqmc_iteration(
            self.expk,
            self.nu,
            self.config,
            self.bmats_up,
            self.bmats_dn,
            self._gf_up,
            self._gf_dn,
            self._sgndet,
            self._logdet,
            self.num_recomp,
            self.prod_len
        )
        # Compute and save acceptance ratio
        acc_ratio = accepted / self.config.size
        self.acceptance_probs.append(acc_ratio)
        logger.debug("[%s] %3d Ratio: %.2f  Signs: (%+d %+d)",
                     self.status, self.it, acc_ratio, self._sgndet[0], self._sgndet[1])

    def accumulate_measurements(self):
        if self.sampl_recomp:
            # Recompute Green's functions before measurements
            self.compute_greens()

        # Accumulate measurements of default observables
        accumulate_measurements(
            self._gf_up,
            self._gf_dn,
            self._sgndet,
            self.n_up,
            self.n_dn,
            self.n_double,
            self.local_moment
        )

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
            self.accumulate_measurements()
            # user measurement callback
            if callback is not None:
                gf_up, gf_dn = self.get_greens()
                out += callback(gf_up, gf_dn, *args, **kwargs)
            self.it += 1
        # Normalize observables
        self.n_up /= sweeps
        self.n_dn /= sweeps
        self.n_double /= sweeps
        self.local_moment /= sweeps
        out /= sweeps
        return out

    def simulate(self, num_equil, num_sampl, callback=None, *args, **kwargs):
        total_sweeps = num_equil + num_sampl
        t0 = time.perf_counter()

        logger.info("Running %s equilibrium sweeps...", num_equil)
        t0_equil = time.perf_counter()
        self.warmup(num_equil)
        t_equil = time.perf_counter() - t0_equil

        logger.info("Running %s sampling sweeps...", num_sampl)
        t0_sampl = time.perf_counter()
        results = self.measure(num_sampl, callback, *args, **kwargs)
        t_sampl = time.perf_counter() - t0_sampl

        t = time.perf_counter() - t0
        logger.info("%s iterations completed!", total_sweeps)
        logger.info("    Signs: [%+d  %+d]", self._sgndet[0], self._sgndet[1])
        logger.info(" Log Dets: [%.2f  %.2f]", self._logdet[0], self._logdet[1])
        logger.info("Equil CPU time: %6.1fs  (%.4f s/it)", t_equil, t_equil / num_equil)
        logger.info("Sampl CPU time: %6.1fs  (%.4f s/it)", t_sampl, t_sampl / num_sampl)
        logger.info("Total CPU time: %6.1fs  (%.4f s/it)", t, t / total_sweeps)
        return results


def run_dqmc(p, callback=None):
    model = hubbard_hypercube(p.shape, p.u, p.eps, p.t, p.mu, p.beta, periodic=True)
    dqmc = DQMC(model, p.num_timesteps, p.num_wraps, p.prod_len, p.seed,
                p.sampl_recomp)
    try:
        extra_results = dqmc.simulate(p.num_equil, p.num_sampl, callback=callback)
    except np.linalg.LinAlgError:
        return ()
    return dqmc.n_up, dqmc.n_dn, dqmc.n_double, dqmc.local_moment, extra_results


def log_parameters(p):
    logger.info("_" * 60)
    logger.info("Simulation parameters")
    logger.info("")
    logger.info("     Shape: %s", p.shape)
    logger.info("         U: %s", p.u)
    logger.info("         t: %s", p.t)
    logger.info("       eps: %s", p.eps)
    logger.info("        mu: %s", p.mu)
    logger.info("      beta: %s", p.beta)
    logger.info("      temp: %s", 1 / p.beta)
    logger.info(" time-step: %s", p.dt)
    logger.info("         L: %s", p.num_timesteps)
    logger.info("    nwraps: %s", p.num_wraps)
    logger.info("   prodLen: %s", p.prod_len)
    logger.info("    nequil: %s", p.num_equil)
    logger.info("    nsampl: %s", p.num_sampl)
    logger.info("    recomp: %s", p.sampl_recomp)
    logger.info("      seed: %s", p.seed)
    logger.info("")


def log_results(*results):
    n_up = np.mean(results[0])
    n_dn = np.mean(results[1])
    n_double = np.mean(results[2])
    local_moment = np.mean(results[3])

    logger.info("_" * 60)
    logger.info("Simulation results")
    logger.info("")
    logger.info("     Total density: %8.4f", n_up + n_dn)
    logger.info("   Spin-up density: %8.4f", n_up)
    logger.info(" Spin-down density: %8.4f", n_dn)
    logger.info("  Double occupancy: %8.4f", n_double)
    logger.info("      Local moment: %8.4f", local_moment)
    if results[4]:
        logger.info("   Callback results: %s", results[4])
    logger.info("")
