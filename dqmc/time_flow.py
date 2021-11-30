# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""This module contains methods for handling the time-step m,atrices from ref [1]_

Notes
-----
The time slice index is called `t` instead of the `l` of the reference.

References
----------
.. [1] Z. Bai et al., “Numerical Methods for Quantum Monte Carlo Simulations
       of the Hubbard Model”, in Series in Contemporary Applied Mathematics,
       Vol. 12 (June 2009), p. 1.
"""

import numpy as np
from numba import njit, float64, int8, int64, void
from numba import types as nt
from .config import UP, DN


@njit(float64[:, :](float64[:, :], float64, int8[:, :], int64, int64))
def compute_timestep_mat(exp_k, nu, config, t, sigma):
    r"""Computes the time step matrix :math:'B_σ(h_t)'.

    Notes
    -----
    The time step matrix :math:'B_σ(h_t)' is defined as
    ..math::
        B_σ(h_t) = e^k e^{σ ν V_t(h_t)}

    Simply multiplying the matrix exponential of the kinetic Hamiltonian with the diagonal
    elements of the second matrix yields the same result as using `np.dot` with `np.diag`.

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
    b : (N, N) np.ndarray
        The matrix :math:'B_{t, σ}(h_t)'.
    """
    return exp_k * np.exp(sigma * nu * config[:, t])


@njit(nt.UniTuple(float64[:, :, :], 2)(float64[:, :], float64, int8[:, :]))
def compute_timestep_mats(exp_k, nu, config):
    r"""Computes the time step matrices :math:'B_σ(h_t)' for all times `t` and both spins.

    Notes
    -----
    The time step matrix :math:'B_σ(h_t)' is defined as
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

    Returns
    -------
    bmats_up : (L, N, N) np.ndarray
        The spin-up time step matrices.
    bmats_dn : (L, N, N) np.ndarray
        The spin-down time step matrices.
    """
    num_sites, num_timesteps = config.shape
    bmats_up = np.zeros((num_timesteps, num_sites, num_sites), dtype=np.float64)
    bmats_dn = np.zeros((num_timesteps, num_sites, num_sites), dtype=np.float64)
    for t in range(num_timesteps):
        bmats_up[t] = compute_timestep_mat(exp_k, nu, config, t, sigma=UP)
        bmats_dn[t] = compute_timestep_mat(exp_k, nu, config, t, sigma=DN)
    return np.ascontiguousarray(bmats_up), np.ascontiguousarray(bmats_dn)


@njit(void(float64[:, :], float64, int8[:, :], float64[:, :, :], float64[:, :, :], int64))
def update_timestep_mats(exp_k, nu, config, bmats_up, bmats_dn, t):
    r"""Updates one time step matrices :math:'B_σ(h_t)' for one time step.

    Parametersc
    ----------
    exp_k : (N, N) np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    bmats_up : (L, N, N) np.ndarray
        The spin-up time step matrices.
    bmats_dn : (L, N, N) np.ndarray
        The spin-down time step matrices.
    t : int
        The index of the time step matrix to update.
    """
    bmats_up[t] = compute_timestep_mat(exp_k, nu, config, t, sigma=UP)
    bmats_dn[t] = compute_timestep_mat(exp_k, nu, config, t, sigma=DN)


@njit(float64[:, :](float64[:, :, :], int64[:]))
def compute_timeflow_map(bmats, order):
    r"""Computes the fermion time flow map matrix :math:'A_σ(h)'.

    Notes
    -----
    The matrix :math:'A_σ' is defined as
    ..math::
        A_σ = B_σ(1) B_σ(2) ... B_σ(L)

    Parameters
    ----------
    bmats : (L, N, N) np.ndarray
        The time step matrices.
    order : np.ndarray, optional
        The order used for multiplying the :math:'B' matrices.
        The default the ascending order, starting from the first time step.

    Returns
    -------
    a : (N, N) np.ndarray
        The time flow map matrix :math:'A_{σ}(h)'.
    """
    if order is None:
        order = np.arange(len(bmats))
    # First matrix
    b_prod = bmats[order[0]]
    # Following matrices multiplied with dot-product
    for i in order[1:]:
        b_prod = np.dot(b_prod, bmats[i])
    return b_prod


@njit  # (nt.UniTuple(float64[:, :, :], 2)(float64[:, :, :], float64[:, :, :], nt.Omitted(int64[:])))
def compute_m_matrices(bmats_up, bmats_dn, order=None):
    r"""Computes the matrix :math:'M_σ = I + A_σ(h)' for both spins.

    Parameters
    ----------
    bmats_up : (L, N, N) np.ndarray
        The spin-up time step matrices.
    bmats_dn : (L, N, N) np.ndarray
        The spin-down time step matrices.
    order : np.ndarray, optional
        The order used for multiplying the :math:'B' matrices.
        The default the ascending order, starting from the first time step.

    Returns
    -------
    m_up : (N, N) np.ndarray
        The spin-up :math:'M' matrix.
    m_dn : (N, N) np.ndarray
        The spin-down :math:'M' matrix.
    """
    if order is None:
        order = np.arange(len(bmats_up), dtype=np.int64)
    eye = np.eye(bmats_up[0].shape[0])
    m_up = eye + compute_timeflow_map(bmats_up, order)
    m_dn = eye + compute_timeflow_map(bmats_dn, order)
    return m_up, m_dn
