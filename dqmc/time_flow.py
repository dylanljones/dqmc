# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from numba import njit
from .config import UP, DN


@njit
def compute_time_step_matrix(exp_k, nu, config, t, sigma):
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


@njit
def compute_time_step_matrices(exp_k, nu, config, sigma):
    r"""Computes the time step matrices :math:'B_σ(h_t)' for all times `t`.

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
    sigma : int
        The spin σ (-1 or +1).

    Returns
    -------
    bmats : (L, N, N) np.ndarray
        The time step matrices.
    """
    num_sites, num_timesteps = config.shape
    bmats = np.zeros((num_timesteps, num_sites, num_sites))
    for t in range(num_timesteps):
        bmats[t] = exp_k * np.exp(sigma * nu * config[:, t])
    return bmats


@njit
def update_time_step_matrices(exp_k, nu, config, bmats_up, bmats_dn, t):
    r"""Updates one time step matrix :math:'B_σ(h_t)' for one time step.

    Parameters
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
    bmats_up[t] = exp_k * np.exp(UP * nu * config[:, t])
    bmats_dn[t] = exp_k * np.exp(DN * nu * config[:, t])


@njit
def compute_time_flow_map(bmats, order=None):
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
