# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import logging
import random
import numpy as np
import scipy.linalg as la
from .config import Configuration, UP, DN

logger = logging.getLogger(__file__)


class LQMC:

    def __init__(self, model, beta, num_times=20):
        self.model = model
        self.config = Configuration.random(model.num_sites, num_timesteps=num_times)

        # Initialize static variables
        self.dtau = beta / self.config.num_timesteps
        self.lamb = np.arccosh(np.exp(self.model.u * self.dtau / 2.)) if model.u else 0
        self.exp_k = la.expm(-self.dtau * model.hamiltonian_kinetic())
        self.gf_up = None
        self.gf_dn = None

        # Check time step size
        check_val = self.model.u * self.model.hop * self.dtau ** 2
        if check_val > 0.1:
            logger.warning("Check-value %.2f should be smaller than 0.1!", check_val)

        self._status = ""

    @property
    def num_sites(self):
        return self.config.num_sites

    @property
    def num_timesteps(self):
        return self.config.num_timesteps

    def compute_expv(self, time, sigma):
        r"""Computes the matrix exponential of the interaction matrix :math:'v_{\sigma}(l)'."""
        diag = sigma * self.lamb * self.config[:, time]
        return np.diagflat(np.exp(-diag))

    def compute_m(self, sigma):
        # compute A=prod(B_l)
        b_prod = 1
        for time in range(self.config.num_timesteps):
            exp_v = self.compute_expv(time, sigma)
            b = np.dot(self.exp_k, exp_v)
            b_prod = np.dot(b_prod, b)
        # Assemble M=I+prod(B)
        return np.eye(self.config.num_sites) + b_prod

    def init_greens(self):
        m_up = self.compute_m(UP)
        m_dn = self.compute_m(DN)
        self.gf_up = la.inv(m_up)
        self.gf_dn = la.inv(m_dn)

    def update_greens(self, site, time):
        arg = 2 * self.lamb * self.config[site, time]
        exp_p = np.exp(+arg)
        exp_m = np.exp(-arg)
        # Update Greens function
        c_up = -(exp_m - 1) * self.gf_up[site, :]
        c_dn = -(exp_p - 1) * self.gf_dn[site, :]
        c_up[site] += (exp_m - 1)
        c_dn[site] += (exp_p - 1)
        b_up = self.gf_up[:, site] / (1 + c_up[site])
        b_dn = self.gf_dn[:, site] / (1 + c_dn[site])
        for j in range(self.num_sites):
            for k in range(self.num_sites):
                self.gf_up[j, k] = self.gf_up[j, k] - b_up[j] * c_up[k]
                self.gf_dn[j, k] = self.gf_dn[j, k] - b_dn[j] * c_dn[k]

    def wrap_greens(self, time):
        expv_up = self.compute_expv(time, sigma=UP)
        expv_dn = self.compute_expv(time, sigma=DN)
        b_up = np.dot(expv_up, self.exp_k)
        b_dn = np.dot(expv_dn, self.exp_k)
        self.gf_up = np.dot(np.dot(b_up, self.gf_up), np.linalg.inv(b_up))
        self.gf_dn = np.dot(np.dot(b_dn, self.gf_dn), np.linalg.inv(b_dn))

    def update_step(self):
        total = self.num_sites * self.num_timesteps
        accepted = 0
        # Iterate over all time-steps, starting at the end (.math:'\beta')
        sites = np.arange(self.num_sites)
        for time in range(self.num_timesteps):
            # Iterate over all lattice sites in a random order
            # np.random.shuffle(sites)
            for site in sites:
                # Compute acceptance ratio
                arg = 2 * self.lamb * self.config[site, time]
                d_up = 1 + (1 - self.gf_up[site, site]) * (np.exp(-arg) - 1)
                d_dn = 1 + (1 - self.gf_dn[site, site]) * (np.exp(+arg) - 1)
                d = d_up * d_dn
                if random.random() < d:
                    accepted += 1
                    # Update HS field and interaction matrices
                    # Update Greens function
                    self.config.update(site, time)
                    self.update_greens(site, time)
            self.wrap_greens(time)

        logger.debug("[%s] Acceptance ratio: %.2f", self._status, accepted / total)

    def warmup_loop(self, sweeps=200):
        self.init_greens()
        self._status = "warmup"
        for it in range(sweeps):
            self.update_step()

    def measure_loop(self, func=None, sweeps=1000):
        self._status = "measurement"
        out = 0
        for it in range(sweeps):
            self.update_step()
            # Callback method for measuring expectation values
            out += func(self.gf_up, self.gf_dn)
        # Normalize output
        return out / sweeps
