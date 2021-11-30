# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Implementations of determinant QMC (DQMC) following ref [1]_

Notes
-----
The time slice index is called `t` instead of the `l` of the reference.

References
----------
.. [1] Z. Bai et al., “Numerical Methods for Quantum Monte Carlo Simulations
       of the Hubbard Model”, in Series in Contemporary Applied Mathematics,
       Vol. 12 (June 2009), p. 1.
"""

import math
import random
import logging
import numpy as np
import scipy.linalg as la
from abc import ABC, abstractmethod
from numba import jit, njit, float64, int8, int64
from numba import types as nt
from .model import HubbardModel
from .config import init_configuration, update_configuration, UP, DN
from .time_flow import compute_timestep_mats, compute_m_matrices, update_timestep_mats
logger = logging.getLogger("dqmc")

RNG = np.random.default_rng(0)

matf64 = float64[:, :]
mati8 = int8[:, :]
tenf64 = float64[:, :, :]


def init_qmc(model, num_timesteps):
    r"""Initializes configuration and static variables of the QMC algorithm.

    Parameters
    ----------
    model : HubbardModel
        The model instance.
    num_timesteps : int
        The number of time steps `L` used in the Monte Carlo simulation.

    Returns
    -------
    exp_k : np.ndarray
        The matrix exponential of the kinetic Hamiltonian of the model.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'.
    config : (N, L) np.ndarray
        The array representing the configuration or or Hubbard-Stratonovich field.
    """
    # Build quadratic terms of Hamiltonian
    ham_k = model.hamiltonian_kinetic()

    # Compute and check time step size
    dtau = model.beta / num_timesteps
    check = model.u * model.hop * dtau ** 2
    if check > 0.1:
        logger.warning("Increase number of time steps: Check-value %.2f should be <0.1!", check)
    else:
        logger.debug("Check-value %.2f is <0.1!", check)

    # Compute factor and matrix exponential of kinetic hamiltonian
    nu = math.acosh(math.exp(model.u * dtau / 2.)) if model.u else 0
    logger.debug("nu=%s", nu)

    exp_k = la.expm(dtau * ham_k)
    logger.debug("exp_k=%s", exp_k)

    # Initialize configuration with random -1 and +1
    config = init_configuration(model.num_sites, num_timesteps)

    return exp_k, nu, config


class BaseDQMC(ABC):

    def __init__(self, model, num_timesteps, time_dir=+1):
        # Init QMC variables
        self.exp_k, self.nu, self.config = init_qmc(model, num_timesteps)

        # Set up time direction and order of inner loops
        self.time_order = np.arange(self.config.shape[1])[::time_dir]
        self.times = np.arange(self.config.shape[1])[::time_dir]
        self.sites = np.arange(self.config.shape[0])

        # Pre-compute time flow matrices
        self.bmats_up, self.bmats_dn = compute_timestep_mats(self.exp_k, self.nu, self.config)

        # Initialize QMC statistics
        self.status = ""
        self.acceptance_probs = list()

        # Initialization callback
        self.initialize()

    def initialize(self):
        pass

    @abstractmethod
    def iteration(self):
        pass

    def greens(self):
        m_up, m_dn = compute_m_matrices(self.bmats_up, self.bmats_dn, self.time_order)
        return la.inv(m_up), la.inv(m_dn)

    def get_greens(self):
        return self.greens()

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
                if callback is not None:
                    gf_up, gf_dn = self.greens()
                    out += callback(gf_up, gf_dn)
        return out / measure


# =========================================================================
# Determinant implementation
# =========================================================================


def iteration_det(exp_k, nu, config, bmats_up, bmats_dn, old_det, times):
    r"""Runs one iteration of the determinant DQMC-scheme.

    Parameters
    ----------
    exp_k : np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    bmats_up : (L, N, N) np.ndarray
        The spin-up time step matrices.
    bmats_dn : (L, N, N) np.ndarray
        The spin-down time step matrices.
    old_det : float
        The old determinant product from the last iteration.
    times : (L,) np.ndarray
        An array of time indices.

    Returns
    -------
    old_det : float
        The last computed determinant product.
    accepted : int
        The number of accepted spin flips.
    """
    accepted = 0
    sites = np.arange(config.shape[0])
    # Iterate over all time-steps
    for t in times:
        # Iterate over all lattice sites
        np.random.shuffle(sites)
        for i in sites:
            # Propose update by flipping spin in confguration
            config[i, t] *= -1
            update_timestep_mats(exp_k, nu, config, bmats_up, bmats_dn, t)
            # Compute determinant product of the new configuration
            m_up, m_dn = compute_m_matrices(bmats_up, bmats_dn, times)
            det_up = np.linalg.det(m_up)
            det_dn = np.linalg.det(m_dn)
            new_det = det_up * det_dn
            # Compute acceptance ratio
            d = min(abs(new_det / old_det), 1.0)
            # Check if move is accepted
            accept = random.random() < d
            if accept:
                # Move accepted: Continue using the new configuration
                accepted += 1
                old_det = new_det
            else:
                # Move not accepted: Revert to the old configuration by updating again
                config[i, t] *= -1
                update_timestep_mats(exp_k, nu, config, bmats_up, bmats_dn, t)

    return old_det, accepted


# =========================================================================
# Rank-1 update implementation
# =========================================================================


@njit(nt.float64(float64, mati8, matf64, matf64, int64, int64))
def compute_acceptance_fast(nu, config, gf_up, gf_dn, i, t):
    r"""Computes the Metropolis acceptance via the fast update scheme.

    Notes
    -----
    A spin-flip of site i and time t is accepted, if d<r:
    ..math::
        α_σ = e^{-2 σ ν s(i, t)} - 1
        d_σ = 1 + (1 - G_{ii, σ}) α_σ
        d = d_↑ d_↓

    Parameters
    ----------
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    gf_up : np.ndarray
        The spin-up Green's function.
    gf_dn : np.ndarray
        The spin-down Green's function.
    i : int
        The site index :math:'i' of the proposed spin-flip.
    t : int
        The time-step index :math:'t' of the proposed spin-flip.

    Returns
    -------
    d : float
        The Metropolis acceptance ratio.
    """
    arg = -2 * nu * config[i, t]
    alpha_up = np.expm1(UP * arg)
    alpha_dn = np.expm1(DN * arg)
    d_up = 1 + alpha_up * (1 - gf_up[i, i])
    d_dn = 1 + alpha_dn * (1 - gf_dn[i, i])
    return min(abs(d_up * d_dn), 1.)


@njit(nt.UniTuple(matf64, 2)(float64, mati8, matf64, matf64, int64, int64))
def update_greens(nu, config, gf_up, gf_dn, i, t):
    r"""Updates the Green's function after accepting a spin-flip.

    Notes
    -----
    The update of the Green's function after the spin at
    site i and time t has been flipped  is defined as
    ..math::
        α_σ = e^{-2 σ ν s(i, t)} - 1
        c_{j,σ} = -α_σ G_{ji,σ} + δ_{ji} α_σ
        b_{k,σ} = G_{ki,σ} / (1 + c_{i,σ})
        G_{jk,σ} = G_{jk,σ} - b_{j,σ}c_{k,σ}

    Parameters
    ----------
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    i : int
        The site index :math:'i' of the proposed spin-flip.
    t : int
        The time-step index :math:'t' of the proposed spin-flip.
    gf_up : np.ndarray
        The spin-up Green's function.
    gf_dn : np.ndarray
        The spin-down Green's function.
    """
    # Compute alphas
    arg = -2 * nu * config[i, t]
    alpha_up = np.expm1(UP * arg)
    alpha_dn = np.expm1(DN * arg)
    # Compute c-vectors for all j
    c_up = -alpha_up * gf_up[i, :]
    c_dn = -alpha_dn * gf_dn[i, :]
    # Add diagonal elements where j=i
    c_up[i] += alpha_up
    c_dn[i] += alpha_dn
    # Compute b-vectors for all k
    b_up = gf_up[:, i] / (1 + c_up[i])
    b_dn = gf_dn[:, i] / (1 + c_dn[i])
    # Compute outer product of b and c and update GF for all j and k
    gf_up -= np.outer(b_up, c_up)
    gf_dn -= np.outer(b_dn, c_dn)
    return gf_up, gf_dn


@njit(nt.UniTuple(matf64, 2)(tenf64, tenf64, matf64, matf64, int64))
def wrap_greens(bmats_up, bmats_dn, gf_up, gf_dn, t):
    r"""Wraps the Green's functions between the time step :math:'t' and :math:'t+1'.

    Parameters
    ----------
    bmats_up : (L, N, N) np.ndarray
        The spin-up time step matrices.
    bmats_dn : (L, N, N) np.ndarray
        The spin-down time step matrices.
    gf_up : (N, N) np.ndarray
        The spin-up Green's function.
    gf_dn : (N. N) np.ndarray
        The spin-down Green's function.
    t : int
        The time-step index :math:'t' of the last iteration over all sites.
    """
    b_up = bmats_up[t]
    b_dn = bmats_dn[t]
    gf_up[:] = np.dot(np.dot(b_up, gf_up), np.linalg.inv(b_up))
    gf_dn[:] = np.dot(np.dot(b_dn, gf_dn), np.linalg.inv(b_dn))
    return gf_up, gf_dn


@njit(nt.Tuple((matf64, matf64, int64))(matf64, float64, mati8, tenf64, tenf64, matf64, matf64, int64[:]))
def iteration_fast(exp_k, nu, config, bmats_up, bmats_dn, gf_up, gf_dn, times):
    r"""Runs one iteration of the rank-1 DQMC-scheme.

    Parameters
    ----------
    exp_k : np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    bmats_up : (L, N, N) np.ndarray
        The spin-up time step matrices.
    bmats_dn : (L, N, N) np.ndarray
        The spin-down time step matrices.
    gf_up : (N, N) np.ndarray
        The spin-up Green's function.
    gf_dn : (N. N) np.ndarray
        The spin-down Green's function.
    times : (L,) np.ndarray
        An array of time indices.

    Returns
    -------
    gf_up : np.ndarray
        The spin-up Green's function after the warmup loop.
    gf_dn : np.ndarray
        The spin-down Green's function after the warmup loop.
    accepted : int
        The number of accepted spin flips.
    """
    accepted = 0
    sites = np.arange(config.shape[0])
    # Iterate over all time-steps
    for t in times:
        # Iterate over all lattice sites randomly
        np.random.shuffle(sites)
        for i in sites:
            # Propose update by flipping spin in confguration
            d = compute_acceptance_fast(nu, config, gf_up, gf_dn, i, t)
            # Check if move is accepted
            accept = random.random() < d
            if accept:
                gf_up, gf_dn = update_greens(nu, config, gf_up, gf_dn, i, t)
                # Move accepted: Continue using the new configuration
                accepted += 1
                config[i, t] = - config[i, t]
                update_timestep_mats(exp_k, nu, config, bmats_up, bmats_dn, t)
        # Wrap Green's function between time steps
        gf_up, gf_dn = wrap_greens(bmats_up, bmats_dn, gf_up, gf_dn, t)

    return gf_up, gf_dn, accepted
