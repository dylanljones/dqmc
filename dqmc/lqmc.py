# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import logging
import random
import numpy as np
import math
from numpy import emath
import scipy.linalg as la
from .config import Configuration, UP, DN

logger = logging.getLogger("dqmc")


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
            logger.warning("Increase number of time steps: Check-value %.2f should be <0.1!",
                           check_val)

        self._status = ""

    @property
    def num_sites(self):
        return self.config.num_sites

    @property
    def num_timesteps(self):
        return self.config.num_timesteps

    def compute_expv(self, time, sigma):
        r"""Computes the matrix exponential of the interaction matrix :math:'v_σ(t)'."""
        diag = sigma * self.lamb * self.config[:, time]
        return np.diagflat(np.exp(diag))

    def compute_m(self, sigma):
        """Compute the matrix :math:'M_σ' used in the inversion for the Green's function.

        Parameters
        ----------
        sigma : int
            The spin σ of the matrix (-1 or +1).

        Notes
        -----
        The matrix :math:'M_σ' is defined as
        ..math::
            M_σ = I + B_σ(1) B_σ(2) ... B_σ(N)
            B_σ(t) = e^k e^{v_σ(t)}
        """
        times = list(range(self.config.num_timesteps))
        # First matrix
        exp_v = self.compute_expv(times[0], sigma)
        b_prod = np.dot(self.exp_k, exp_v)
        # Following matrices multiplied with dot-product
        for time in times[1:]:
            exp_v = self.compute_expv(time, sigma)
            b = np.dot(self.exp_k, exp_v)
            b_prod = np.dot(b_prod, b)

        return np.eye(self.config.num_sites) + b_prod

    def compute_greens(self):
        """Computes the Green's functions for both spins via matrix inversion."""
        m_up = self.compute_m(UP)
        m_dn = self.compute_m(DN)
        self.gf_up = la.inv(m_up)
        self.gf_dn = la.inv(m_dn)

    def update_greens(self, site: int, time: int) -> None:
        r"""Updates the Green's function after accepting a spin-flip.

        Notes
        -----
        The update of the Green's function after the spin at
        site i and time t has been flipped  is defined as
        ..math::
            α_↑ = e^{-2 λ s(i, t)} - 1, α_↓ = e^{+2 λ s(i, t)} - 1
            c_{j,σ} = -α_σ G_{ji,σ} + δ_{ji} α_σ
            b_{k,σ} = G_{ki,σ} / (1 + c_{i,σ})
            G_{jk,σ} = G_{jk,σ} - b_{j,σ}c_{k,σ}

        Parameters
        ----------
        site : int
            The index of the site of the flipped spin.
        time : int
            The index of the time step of the flipped spin.
        """
        # Compute alphas
        arg = 2 * self.lamb * self.config[site, time]
        alpha_up = (np.exp(-arg) - 1)
        alpha_dn = (np.exp(+arg) - 1)
        # Compute c-vectors for all j
        c_up = -alpha_up * self.gf_up[site, :]
        c_dn = -alpha_dn * self.gf_dn[site, :]
        # Add diagonal elements where j=i
        c_up[site] += alpha_up
        c_dn[site] += alpha_dn
        # Compute b-vectors for all k
        b_up = self.gf_up[:, site] / (1 + c_up[site])
        b_dn = self.gf_dn[:, site] / (1 + c_dn[site])
        # Compute outer product of b and c and update GF for all j and k
        self.gf_up += -np.dot(b_up[:, None], c_up[None, :])
        self.gf_dn += -np.dot(b_dn[:, None], c_dn[None, :])

    def wrap_greens(self, time):
        """Wraps the Green's functions between two time steps."""
        expv_up = self.compute_expv(time, sigma=UP)
        expv_dn = self.compute_expv(time, sigma=DN)
        b_up = np.dot(self.exp_k, expv_up)
        b_dn = np.dot(self.exp_k, expv_dn)
        self.gf_up = np.dot(np.dot(b_up, self.gf_up), la.inv(b_up))
        self.gf_dn = np.dot(np.dot(b_dn, self.gf_dn), la.inv(b_dn))

    def update_step(self):
        """Performs one sweep of the configuration over all time steps and sites.

        Notes
        -----
        A spin-flip of site i and time t is accepted, if d<r:
        ..math::
            α_↑ = e^{-2 λ s(i, t)} - 1, α_↓ = e^{+2 λ s(i, t)} - 1
            d_σ = 1 + (1 - G_{ii, σ}) α_σ
            d = d_↑ d_↓
        """
        total = self.num_sites * self.num_timesteps
        accepted = 0
        # Iterate over all time-steps, starting at the end (.math:'\beta')
        sites = np.arange(self.num_sites)
        for time in reversed(range(self.num_timesteps)):
            # Iterate over all lattice sites
            # np.random.shuffle(sites) #  in a random order
            for site in sites:
                # Compute acceptance ratio
                arg = 2 * self.lamb * self.config[site, time]
                d_up = 1 + (1 - self.gf_up[site, site]) * (np.exp(-arg) - 1)
                d_dn = 1 + (1 - self.gf_dn[site, site]) * (np.exp(+arg) - 1)
                d = d_up * d_dn
                if random.random() < d:
                    accepted += 1
                    # Update HS field and interaction matrices
                    self.config.update(site, time)
                    # Update Greens function
                    self.update_greens(site, time)
            self.wrap_greens(time)
        logger.debug("[%s] Acceptance ratio: %.2f", self._status, accepted / total)

    def warmup_loop(self, sweeps=200):
        """Runs the warmup loop to (hopefully) settle the system near a equilibrium state.

        Parameters
        ----------
        sweeps : int, optional
            The number of full sweeps of the configuration array (hundreds).
        """
        self._status = "warmup"
        # Initialize Green's functions
        self.compute_greens()
        # Perform warmup sweeps
        for it in range(sweeps):
            self.update_step()

    def measure_loop(self, func, sweeps=1000):
        """Runs the measurement loops and performs measurements after each sweep.

        Parameters
        ----------
        func : callable
            A function called after every sweep to compute expectation values.
            The parameters are the up and down Green's function. If the result has
            the form of a sequence a 'np.ndarray' has to be returned to ensure
            the final normalization of the results.
        sweeps : int, optional
            The number of full sweeps of the configuration array (thousands).

        Returns
        -------
        res : float or np.ndarray
            The one or more measured epectation values.
        """
        self._status = "measurement"
        out = 0
        for it in range(sweeps):
            self.update_step()
            # Callback method for measuring expectation values
            out += func(self.gf_up, self.gf_dn)
        # Normalize output
        return out / sweeps
