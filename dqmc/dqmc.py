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

import random
import logging
import numpy as np
import scipy.linalg as la
from numba import jit
from .model import HubbardModel
from .config import init_configuration, update_configuration, UP, DN

logger = logging.getLogger("dqmc")

RNG = np.random.default_rng(0)


def iter_times(config, reverse=False):
    items = range(config.shape[1])
    return reversed(items) if reverse else items


def iter_sites(config, rand=False):
    sites = np.arange(config.shape[0])
    if rand:
        RNG.shuffle(sites)
    return sites


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
    config : (N, L) np.ndarray
        The array representing the configuration or or Hubbard-Stratonovich field.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'.
    exp_k : np.ndarray
        The matrix exponential of the kinetic Hamiltonian of the model.
    """
    # Compute and check time step size
    dtau = model.beta / num_timesteps
    check = model.u * model.hop * dtau ** 2
    if check > 0.1:
        logger.warning("Increase number of time steps: Check-value %.2f should be <0.1!", check)
    else:
        logger.debug("Check-value %.2f is <0.1!", check)
    # Initialize configuration
    config = init_configuration(model.num_sites, num_timesteps)
    # Compute factor and matrix exponential of kinetic hamiltonian
    nu = np.arccosh(np.exp(model.u * dtau / 2.)) if model.u else 0
    exp_k = la.expm(dtau * model.hamiltonian_kinetic())

    return config, nu, exp_k


@jit(nopython=True)
def bmatrix(exp_k, nu, config, t, sigma):
    r"""Computes the matrix :math:'B_σ(h_t)'.

    Notes
    -----
    The matrix :math:'B_σ(h_t)' is defined as
    ..math::
        B_σ(h_t) = e^k e^{σ ν V_t(h_t)}

    Parameters
    ----------
    exp_k : (N, N) np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    t : int
        The index of the time step.
    sigma : int
        The spin σ (-1 or +1).

    Returns
    -------
    b : np.ndarray
        The matrix :math:'B_{t, σ}(h_t)'.
    """
    diag = np.exp(sigma * nu * config[:, t])
    # return np.dot(exp_k, np.diag(diag))
    return exp_k * diag


@jit(nopython=True)
def mmatrix(exp_k, nu, config, sigma):
    r"""Computes the fermion matrix :math:'M_σ(h)'.

    Notes
    -----
    The matrix :math:'M_σ' is defined as
    ..math::
        M_σ = I + B_σ(L) B_σ(L-1) ... B_σ(1)
        B_σ(h_t) = e^k e^{σ ν V_t(h_t)}

    Parameters
    ----------
    exp_k : np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    sigma : int
        The spin σ (-1 or +1).

    Returns
    -------
    m : np.ndarray
        The fermion matrix :math:'M_{σ}(h)'.
    """
    # Initialize the time indices of the B-matrices (starts with the last time step)
    times = np.arange(config.shape[1])

    # First matrix
    b_prod = bmatrix(exp_k, nu, config, times[0], sigma)
    # Following matrices multiplied with dot-product
    for t in times[1:]:
        b = bmatrix(exp_k, nu, config, t, sigma)
        b_prod = np.dot(b_prod, b)
    # Add identity matrix
    return np.eye(config.shape[0]) + b_prod


def sample_acceptance(d):
    """Helper function for accepting a proposed update via the Metropolis acceptance ratio.

    Parameters
    ----------
    d : float
        The Metropolis acceptance ratio.

    Returns
    -------
    accepted : bool
        Flag if the proposed update is accepted or not.
    """
    return random.random() < d


def compute_det(exp_k, nu, config):
    r"""Computes the product of the spin-up and -down determinants used for the acceptance ratio.

    Notes
    -----
    The product of determinants is defined as
    ..math::
        det[M_↑(h)] det[M_↓(h)]

    Parameters
    ----------
    exp_k : np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.

    Returns
    -------
    det_prod : float
        The product of the spin-up and -down determinants.
    """
    m_up = mmatrix(exp_k, nu, config, sigma=UP)
    m_dn = mmatrix(exp_k, nu, config, sigma=DN)
    return la.det(m_up) * la.det(m_dn)


@jit(nopython=True)
def compute_acceptance_det(old_det, new_det):
    r"""Computes the Metropolis acceptance via the determinants.

    Notes
    -----
    The acceptance is defined as the fraction
    ..math::
        d = det[M_↑(h')] det[M_↓(h')] / (det[M_↑(h)] det[M_↓(h)])

    where :math:' h' ' is the new configuration and :math:' h ' the old one.

    Parameters
    ----------
    old_det : float
        The previous determinant product.
    new_det : float
        The new determinant product.

    Returns
    -------
    d : float
        The Metropolis acceptance ratio.
    """
    return new_det / old_det


@jit(nopython=True)
def compute_acceptance_fast(nu, config, i, t, gf_up, gf_dn):
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
    i : int
        The site index :math:'i' of the proposed spin-flip.
    t : int
        The time-step index :math:'t' of the proposed spin-flip.
    gf_up : np.ndarray
        The spin-up Green's function.
    gf_dn : np.ndarray
        The spin-down Green's function.

    Returns
    -------
    d : float
        The Metropolis acceptance ratio.
    """
    arg = -2 * nu * config[i, t]
    alpha_up = (np.exp(UP * arg) - 1)
    alpha_dn = (np.exp(DN * arg) - 1)
    d_up = 1 + alpha_up * (1 - gf_up[i, i])
    d_dn = 1 + alpha_dn * (1 - gf_dn[i, i])
    return d_up * d_dn


def compute_greens(exp_k, nu, config):
    r"""Computes the spin-up and -down Green's function for the configuration.

    Parameters
    ----------
    exp_k : np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.

    Returns
    -------
    gf_up : np.ndarray
        The spin-up Green's function.
    gf_dn : np.ndarray
        The spin-down Green's function.
    """
    m_up = mmatrix(exp_k, nu, config, sigma=UP)
    m_dn = mmatrix(exp_k, nu, config, sigma=DN)
    gf_up = la.inv(m_up)
    gf_dn = la.inv(m_dn)
    return gf_up, gf_dn


def update_greens(nu, config, i, t, gf_up, gf_dn):
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

    Returns
    -------
    gf_up : np.ndarray
        The updated spin-up Green's function.
    gf_dn : np.ndarray
        The updated spin-down Green's function.
    """
    # Compute alphas
    arg = -2 * nu * config[i, t]
    alpha_up = (np.exp(UP * arg) - 1)
    alpha_dn = (np.exp(DN * arg) - 1)
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
    gf_up += -np.dot(b_up[:, None], c_up[None, :])
    gf_dn += -np.dot(b_dn[:, None], c_dn[None, :])
    return gf_up, gf_dn


def wrap_greens(exp_k, nu, config, t, gf_up, gf_dn):
    r"""Wraps the Green's functions between the time step :math:'t' and :math:'t+1'.

    Parameters
    ----------
    exp_k : np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    t : int
        The time-step index :math:'t' of the last iteration over all sites.
    gf_up : np.ndarray
        The spin-up Green's function.
    gf_dn : np.ndarray
        The spin-down Green's function.

    Returns
    -------
    gf_up : np.ndarray
        The wrapped spin-up Green's function.
    gf_dn : np.ndarray
        The wrapped spin-down Green's function.
    """
    b_up = bmatrix(exp_k, nu, config, t, sigma=UP)
    b_dn = bmatrix(exp_k, nu, config, t, sigma=DN)
    gf_up = np.dot(np.dot(b_up, gf_up), la.inv(b_up))
    gf_dn = np.dot(np.dot(b_dn, gf_dn), la.inv(b_dn))
    return gf_up, gf_dn


# =========================================================================


def warmup_loop_det(exp_k, nu, config, sweeps=200):
    r"""Runs the determinant warmup loop to (hopefully) settle the system near the equilibrium.

    Parameters
    ----------
    exp_k : np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    sweeps : int, optional
        The number of full sweeps of the configuration array (hundreds).

    Returns
    -------
    old_det : float
        The product of determinants of the coonfiguration after the last accepted spin flip.
    """
    # Initialize the determinant product
    old_det = compute_det(exp_k, nu, config)
    # Warmup-sweeps
    for _ in range(sweeps):
        # Iterate over all time-steps
        for t in range(config.shape[1]):
            # Iterate over all lattice sites
            for i in range(config.shape[0]):
                # Propose update by flipping spin in confguration
                update_configuration(config, i, t)
                # Compute determinant product of the new configuration
                new_det = compute_det(exp_k, nu, config)
                # Compute acceptance ratio
                d = compute_acceptance_det(old_det, new_det)
                # Check if move is accepted
                if sample_acceptance(d):
                    # Move accepted: Continue using the new configuration
                    old_det = new_det
                else:
                    # Move not accepted: Revert to the old configuration by updating again
                    update_configuration(config, i, t)
    return old_det


def warmup_loop_fast(exp_k, nu, config, sweeps=200):
    r"""Runs the rank-1 warmup loop to (hopefully) settle the system near the equilibrium.

    Parameters
    ----------
    exp_k : np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    sweeps : int, optional
        The number of full sweeps of the configuration array (hundreds).

    Returns
    -------
    gf_up : np.ndarray
        The spin-up Green's function after the warmup loop.
    gf_dn : np.ndarray
        The spin-down Green's function after the warmup loop.
    """
    # Initialize Green's functions
    m_up = mmatrix(exp_k, nu, config, sigma=UP)
    m_dn = mmatrix(exp_k, nu, config, sigma=DN)
    gf_up = la.inv(m_up)
    gf_dn = la.inv(m_dn)
    # Warmup-sweeps
    for _ in range(sweeps):
        # Iterate over all time-steps
        for t in range(config.shape[1]):
            # Iterate over all lattice sites
            for i in range(config.shape[0]):
                # Compute acceptance ratio
                d = compute_acceptance_fast(nu, config, i, t, gf_up, gf_dn)
                # Check if move is accepted
                if sample_acceptance(d):
                    # Move accepted: update configuration and Green's functions
                    gf_up, gf_dn = update_greens(nu, config, i, t, gf_up, gf_dn)
                    update_configuration(config, i, t)
            # Wrap Green#s function between time steps
            gf_up, gf_dn = wrap_greens(exp_k, nu, config, t, gf_up, gf_dn)
    return gf_up, gf_dn
