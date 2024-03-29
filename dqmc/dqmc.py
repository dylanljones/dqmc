# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Implementations of determinant QMC (DQMC) following ref [1]_

Notes
-----
The time slice index is called `t` instead of the `l` of the reference.
Some methods use just in time compilation (jit) even though no speed-up is expected.
This is done to ensure these methods can be called in other jited methods in no-python
mode.

References
----------
.. [1] Z. Bai et al., “Numerical Methods for Quantum Monte Carlo Simulations
       of the Hubbard Model”, in Series in Contemporary Applied Mathematics,
       Vol. 12 (June 2009), p. 1.
"""

import math
import logging
import numpy as np
import numpy.linalg as la
from scipy.linalg import expm, lapack
from numba import njit, float64, int8, int64, void
from numba import types as nt
from .model import HubbardModel  # noqa: F401
from .linalg import timeflow_map, matrix_product_sequence_0beta, mdot
from .linalg import numpy_dger as dger

# Try to import Fortran implementation
try:
    from .src import construct_greens as _construct_greens_f
    _fortran_available = True
except ImportError:
    _construct_greens_f = None
    _fortran_available = False

logger = logging.getLogger("dqmc")

expk_t = float64[:, :]
conf_t = int8[:, :]
bmat_t = float64[:, :, ::1]
gmat_t = float64[:, ::1]
gtau_t = float64[:, :, ::1]

UP, DN = +1, -1

jkwargs = dict(nogil=True, fastmath=True, cache=True)


def init_configuration(num_sites: int, num_times: int) -> np.ndarray:
    """Initializes the Hubbard-Stratonovich field for the DQMC simulation.

    The 'Hubbard-Stratonovich field' (also called the 'configuration') is an
    auxilliary field of spins, represented by an array filled with a
    random distribution of `-1` (spin down) and `+1` (spin up).

    Parameters
    ----------
    num_sites : int
        The number of sites `N` of the lattice model.
    num_times : int
        The number of time steps `L` used in the Monte Carlo simulation.

    Returns
    -------
    config : (N, L) np.ndarray
        The array representing the configuration or Hubbard-Stratonovich field.
    """
    # samples = random.choices([-1, +1], k=num_sites * num_times)
    samples = np.random.choice([-1, +1], size=num_sites * num_times)
    config = np.array(samples).reshape((num_sites, num_times)).astype(np.int8)
    return config


def init_qmc(model, num_times, seed):
    r"""Initializes the configuration and static variables of the QMC algorithm.

    Initializes the main quantities used in the DQMC simulations. The regular and
    inverse matrix exponentials of the non-interacting Hamiltonian :math:'e^{Δτ H_0}'
    as well as the parameter :math:'ν', defined by
    .. math::
        \cosh(ν) = e^{U Δτ / 2},

    are constant and are reused throughout the simulation. Additionally, the
    Hubbard-Stratonovich field is initialized with random values.

    Parameters
    ----------
    model : HubbardModel
        The model instance.
    num_times : int
        The number of time steps `L` used in the Monte Carlo simulation.
    seed : int
        A seed to be set before creating the cinfiguration.
    Returns
    -------
    expk : np.ndarray
        The matrix exponential of the kinetic Hamiltonian of the model.
    expk_inv : np.ndarray
        The inverse matrix exponential of the kinetic Hamiltonian of the model.
    nu : float
        The parameter ν defined by :math:`\cosh(ν) = e^{U Δτ / 2}`.
    config : (N, L) np.ndarray
        The array representing the configuration or Hubbard-Stratonovich field.

    Notes
    -----
    In order to ensure good simulation results, the length of each imaginary
    time slice :math:`Δτ` should be chosen such that
    .. math::
        t U (Δτ)^2 < 1 / 10

    If the number of imaginary time steps `num_times` is set to low a warning is given.

    See Also
    --------
    init_configuration : Initializes the Hubbard-Stratonovich field.
    """

    np.random.seed(seed)

    # Build quadratic terms of Hamiltonian
    ham_k = model.hamiltonian_kinetic()

    # Compute and check time step size
    dtau = model.beta / num_times
    check = model.u * model.hop * dtau ** 2
    if check > 0.1:
        dt_max = math.sqrt(0.1 / (model.u * model.hop))
        num_times_min = math.ceil(model.beta / dt_max)
        logger.warning(
            "Check-value %.2f should be <0.1, increase number of time steps to >%s!",
            check, num_times_min
        )
    else:
        logger.debug("Check-value %.4f is <0.1!", check)

    # Compute factor and matrix exponential of kinetic hamiltonian
    nu = math.acosh(math.exp(model.u * dtau / 2.0)) if model.u else 0
    logger.debug("nu=%s", nu)

    expk = expm(dtau * ham_k)
    expk_inv = la.inv(expk)
    logger.debug("min(e^k)=%s", np.min(expk))
    logger.debug("max(e^k)=%s", np.max(expk))

    # Initialize configuration with random -1 and +1
    config = init_configuration(model.num_sites, num_times)
    prop_pos = len(np.where(config == +1)[0]) / config.size
    prop_neg = len(np.where(config == -1)[0]) / config.size
    logger.debug("config: p+=%s p-=%s", prop_pos, prop_neg)

    return expk, expk_inv, nu, config


# =========================================================================
# Time flow methods
# =========================================================================


@njit(float64[:, ::1](expk_t, float64, conf_t, int64, int64), **jkwargs)
def compute_timestep_mat(expk, nu, config, t, sigma):
    r"""Computes the time step matrix :math:`B_σ(h_t)`.

    The time step matrix :math:'B_σ(h_t)' is defined as
    .. math::
        B_σ(h_t) = e^{Δτ H_0} e^{σ ν V_t(h_t)},

    where the first term is the matrix exponential of the non-interacting Hamiltonian.

    Parameters
    ----------
    expk : (N, N) np.ndarray
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

    Notes
    -----
    The time step matrix :math:'B_σ(h_t)' can be computed by scaling the rows
    of the matrix exponential of the non-interacting Hamiltonian with the exponential
    of the interaction matrix.
    """
    return expk * np.exp(sigma * nu * config[:, t])


@njit(float64[:, ::1](expk_t, float64, conf_t, int64, int64), **jkwargs)
def compute_timestep_mat_inv(expk_inv, nu, config, t, sigma):
    r"""Computes the inverse time step matrix :math:`B_σ(h_t)^{-1}`.

    The time step matrix :math:`B_σ(h_t)^{-1}` is defined as
    .. math::
        B_σ(h_t)^{-1} = [e^{Δτ H_0} e^{σ ν V_t(h_t)}]^{-1},

    where the first term is the matrix exponential of the non-interacting Hamiltonian.

    Parameters
    ----------
    expk_inv : (N, N) np.ndarray
        The inverse matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:`\cosh(ν) = e^{U Δτ / 2}`
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    t : int
        The index of the time step.
    sigma : int
        The spin σ (-1 or +1).
    Returns
    -------
    b_inv : (N, N) np.ndarray
        The matrix :math:`B_{t, σ}(h_t)^{-1}`.

    Notes
    -----
    The inverse time step matrix :math:'B_σ(h_t)^{-1}' can be computed scaling the
    inverse matrix exponential of the kinetic hamiltonian:
    ..math::
        B_σ(h_t)^{-1} = [e^{Δτ H_0} e^{σ ν V_t(h_t)}]^{-1}
        = [e^{Δτ H_0}]^{-1} e^{-σ ν V_t(h_t)}
    """
    tmp = np.zeros((config.shape[0], 1), dtype=np.float64)
    tmp[:, 0] = np.exp(-sigma * nu * config[:, t])
    return expk_inv * tmp


@njit(nt.UniTuple(bmat_t, 2)(expk_t, float64, conf_t), **jkwargs)
def compute_timestep_mats(expk, nu, config):
    r"""Computes the time step matrices :math:`B_σ(h_t)` for all times and both spins.

    Parameters
    ----------
    expk : (N, N) np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:`\cosh(ν) = e^{U Δτ / 2}`.
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    Returns
    -------
    bmats_up : (L, N, N) np.ndarray
        The spin-up time step matrices.
    bmats_dn : (L, N, N) np.ndarray
        The spin-down time step matrices.

    See Also
    --------
    compute_timestep_mat : Computation of the individual time step matrices.
    """
    num_sites, num_timesteps = config.shape
    bmats_up = np.zeros((num_timesteps, num_sites, num_sites), dtype=np.float64)
    bmats_dn = np.zeros((num_timesteps, num_sites, num_sites), dtype=np.float64)
    for t in range(num_timesteps):
        bmats_up[t] = compute_timestep_mat(expk, nu, config, t, sigma=UP)
        bmats_dn[t] = compute_timestep_mat(expk, nu, config, t, sigma=DN)
    return np.ascontiguousarray(bmats_up), np.ascontiguousarray(bmats_dn)


@njit(nt.UniTuple(bmat_t, 2)(expk_t, float64, conf_t), **jkwargs)
def compute_timestep_mats_inv(expk_inv, nu, config):
    r"""Computes the inverse of the time step matrices :math:`B_σ(h_t)^{-1}`.

    Parameters
    ----------
    expk_inv : (N, N) np.ndarray
        The inverse matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:`\cosh(ν) = e^{U Δτ / 2}`.
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    Returns
    -------
    bmats_inv_up : (L, N, N) np.ndarray
        The inverse spin-up time step matrices.
    bmats_inv_dn : (L, N, N) np.ndarray
        The inverse spin-down time step matrices.

    See Also
    --------
    compute_timestep_mat_inv : Computation of the individual inverse time step matrices.
    """
    num_sites, num_timesteps = config.shape
    bmats_inv_up = np.zeros((num_timesteps, num_sites, num_sites), dtype=np.float64)
    bmats_inv_dn = np.zeros((num_timesteps, num_sites, num_sites), dtype=np.float64)
    for t in range(num_timesteps):
        bmats_inv_up[t] = compute_timestep_mat_inv(expk_inv, nu, config, t, sigma=UP)
        bmats_inv_dn[t] = compute_timestep_mat_inv(expk_inv, nu, config, t, sigma=DN)
    return np.ascontiguousarray(bmats_inv_up), np.ascontiguousarray(bmats_inv_dn)


@njit(void(expk_t, float64, conf_t, bmat_t, bmat_t, int64), **jkwargs)
def update_timestep_mats(expk, nu, config, bmats_up, bmats_dn, t):
    r"""Updates a time step matrices :math:'B_σ(h_t)' for one time step.

    Parameters
    ----------
    expk : (N, N) np.ndarray
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
    bmats_up[t] = compute_timestep_mat(expk, nu, config, t, sigma=UP)
    bmats_dn[t] = compute_timestep_mat(expk, nu, config, t, sigma=DN)


@njit(void(expk_t, float64, conf_t, bmat_t, bmat_t, int64), **jkwargs)
def update_timestep_mats_inv(expk_inv, nu, config, bmats_up_inv, bmats_dn_inv, t):
    r"""Updates a inverse time step matrices :math:'B_σ(h_t)^{-1}' for one time step.

    Parameters
    ----------
    expk_inv : (N, N) np.ndarray
        The inverse matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    bmats_up_inv : (L, N, N) np.ndarray
        The inverse spin-up time step matrices.
    bmats_dn_inv : (L, N, N) np.ndarray
        The inverse spin-down time step matrices.
    t : int
        The index of the time step matrix to update.
    """
    bmats_up_inv[t] = compute_timestep_mat_inv(expk_inv, nu, config, t, sigma=UP)
    bmats_dn_inv[t] = compute_timestep_mat_inv(expk_inv, nu, config, t, sigma=DN)


# =========================================================================
# Green's function methods
# =========================================================================


@njit(nt.UniTuple(float64[:, :], 2)(bmat_t, bmat_t, int64), **jkwargs)
def compute_m_matrices(bmats_up, bmats_dn, t):
    r"""Computes the matrix :math:'M_σ = I + A_σ(h)' for both spins.

    Parameters
    ----------
    bmats_up : (L, N, N) np.ndarray
        The spin-up time step matrices.
    bmats_dn : (L, N, N) np.ndarray
        The spin-down time step matrices.
    t : int
        The current time-step index :math:'t'.
    Returns
    -------
    m_up : (N, N) np.ndarray
        The spin-up :math:'M' matrix.
    m_dn : (N, N) np.ndarray
        The spin-down :math:'M' matrix.
    """
    # Compute product of B-matrices in reverse order,
    # such that the last index is `t`
    order = np.arange(bmats_up.shape[0], 0, -1) - 1
    if t != 0:
        order = np.roll(order, t)

    a_up = bmats_up[order[0]]
    a_dn = bmats_dn[order[0]]
    for i in order[1:]:
        a_up = np.dot(a_up, bmats_up[i])
        a_dn = np.dot(a_dn, bmats_dn[i])

    # Add identities to products
    eye = np.eye(bmats_up[0].shape[0], dtype=np.float64)
    m_up = eye + a_up
    m_dn = eye + a_dn
    return m_up, m_dn


@njit((bmat_t, bmat_t, gmat_t, gmat_t, int64[:], float64[:], int64), **jkwargs)
def compute_greens(bmats_up, bmats_dn, gf_up, gf_dn, sgndet, logdet, t):
    r"""Computes the Green's functions for both spins.

    Parameters
    ----------
    bmats_up : (L, N, N) np.ndarray
        The spin-up time step matrices.
    bmats_dn : (L, N, N) np.ndarray
        The spin-down time step matrices.
    gf_up : (N, N) np.ndarray
        Output array for the spin-up Green's function.
    gf_dn : (N, N) np.ndarray
        Output array for the spin-down Green's function.
    sgndet : (2, ) np.ndarray
        The signs of the determinants of the Green's functions.
    logdet : (2, ) np.ndarray
        The logarithm of the determinants of the Green's functions.
    t : int
        The current time-step index :math:'t'.
    """
    m_up, m_dn = compute_m_matrices(bmats_up, bmats_dn, t)
    gf_up[:, :] = np.linalg.inv(m_up)
    gf_dn[:, :] = np.linalg.inv(m_dn)
    sgndet[0] = np.sign(np.linalg.det(gf_up))
    sgndet[1] = np.sign(np.linalg.det(gf_dn))
    logdet[0] = np.log(np.abs(np.linalg.det(gf_up)))
    logdet[1] = np.log(np.abs(np.linalg.det(gf_dn)))


def _construct_greens(tsm, gf):
    """Construct the Green's function (I + A)^{-1} with the imaginary-time flow map A.

    Parameters
    ----------
    tsm :  : (L, N, N) np.ndarray
        The time step matrices for one spin channel.
    gf : (N, N) np.ndarray
        The output array for the Green's function.
    Returns
    -------
    sign : int
        The sign of the determinant of the Green#s function.

    References
    ----------
    .. [1] Z. Bai et al., “Stable solutions of linear systems involving long chain
           of matrix multiplications”, in Linear Algebra Appl. 435, p. 659-673 (2011)
    """
    # Calculate imaginary time flow map
    q, d, t, tau, lwork = timeflow_map(tsm)

    # Construct the matrix D_b^{-1}
    diagb = np.copy(d)
    maskb = np.abs(diagb) < 1
    diagb[maskb] = 1.0
    db_inv = np.diag(1 / diagb)
    # Calculate D_s T and store result in T
    n = q.shape[0]
    db_inv_qt, work, info = lapack.dormqr("R", "T", q, tau, db_inv, lwork)
    # Calculate D_b^{-1} Q^T + D_s T, store result in T
    for i in range(n):
        if np.abs(d[i]) <= 1.0:
            t[i, :] *= d[i]
    t += db_inv_qt
    # Perform LU decomposition of D_b^{-1} Q^T + D_s T
    t_lu, jpvt, info = lapack.dgetrf(t)
    # Calculate sign/determinant of (D_b^{-1} Q^T + D_s T)^{-1} (D_b^{-1} Q^T)
    sign = 1
    for i in range(n):
        if (t_lu[i, i] < 0) ^ (jpvt[i] != i) ^ (db_inv[i, i] < 0) ^ (tau[i] > 0):
            sign *= -1
    # Calculate (D_b^{-1} Q^T + D_s T)^{-1} (D_b^{-1} Q^T) and overwrite gf
    # DGETRS solves a system of linear equations A * X = B with a general N-by-N
    # matrix A using the LU factorization computed by DGETRF.
    _gf, info = lapack.dgetrs(t_lu, jpvt, db_inv_qt)
    gf[:, :] = _gf
    logdet = np.log(np.abs(np.linalg.det(gf)))
    return sign, logdet


def compute_greens_qrd(bmats_up, bmats_dn, gf_up, gf_dn, sgndet, logdet, t, prod_len=1):
    r"""Computes and overwrites the Green's functions via ASvQRD stabilization.

    Parameters
    ----------
    bmats_up : (L, N, N) np.ndarray
        The spin-up time step matrices.
    bmats_dn : (L, N, N) np.ndarray
        The spin-down time step matrices.
    gf_up : np.ndarray
        Output array for the spin-up Green's function.
    gf_dn : np.ndarray
        Output array for the spin-down Green's function.
    sgndet : (2, ) np.ndarray
        The signs of the determinants of the Green's functions.
    logdet : (2, ) np.ndarray
        The logarithm of the determinants of the Green's functions.
    t : int
        The current time-step index :math:'t'.
    prod_len : int
        The number of matrices multiplied explicitly
    """
    tsm_up = matrix_product_sequence_0beta(bmats_up, prod_len, t)
    tsm_dn = matrix_product_sequence_0beta(bmats_dn, prod_len, t)
    if _fortran_available:
        sgndet[0], logdet[0] = _construct_greens_f(tsm_up, gf_up)
        sgndet[1], logdet[1] = _construct_greens_f(tsm_dn, gf_dn)
    else:
        sgndet[0], logdet[0] = _construct_greens(tsm_up, gf_up)
        sgndet[1], logdet[1] = _construct_greens(tsm_dn, gf_dn)


def init_greens(bmats_up, bmats_dn, t, prod_len=0):
    r"""Initializes and computes the Green's functions naively.

    Parameters
    ----------
    bmats_up : (L, N, N) np.ndarray
        The spin-up time step matrices.
    bmats_dn : (L, N, N) np.ndarray
        The spin-down time step matrices.
    t : int
        The current time-step index :math:'t'.
    prod_len : int
        The number of matrices multiplied explicitly

    Returns
    -------
    gf_up : np.ndarray
        The spin-up Green's function.
    gf_dn : np.ndarray
        The spin-down Green's function.
    sgndet : (2, ) np.ndarray
        The signs of the determinants of the Green's functions.
    logdet : (2, ) np.ndarray
        The logarithm of the determinants of the Green's functions.
    """
    num_sites = bmats_up[0].shape[0]
    shape = (num_sites, num_sites)
    sgndet = np.zeros(2, dtype=np.int64)
    logdet = np.zeros(2, dtype=np.float64)
    gf_up = np.ascontiguousarray(np.zeros(shape, dtype=np.float64))
    gf_dn = np.ascontiguousarray(np.zeros(shape, dtype=np.float64))
    if prod_len > 0:
        compute_greens_qrd(bmats_up, bmats_dn, gf_up, gf_dn, sgndet, logdet, t,
                           prod_len)
    else:
        compute_greens(bmats_up, bmats_dn, gf_up, gf_dn, sgndet, logdet, t)
    return gf_up, gf_dn, sgndet, logdet


def init_gftau(num_sites, num_times):
    gf_up = np.zeros((num_times, num_sites, num_sites), dtype=np.float64)
    gf_dn = np.zeros((num_times, num_sites, num_sites), dtype=np.float64)
    return gf_up, gf_dn


@njit(gtau_t(bmat_t, gmat_t))
def compute_unequal_time_greens(bmats, gf0):
    r"""Computes the unequal-time Green's function :math:'G(τ, 0)'.

    Parameters
    ----------
    bmats : (L, N, N) np.ndarray
        The time step matrices.
    gf0 : (N, N) np.ndarray
        The equal time Green's function 'G_0' for 'τ=0'.
    Returns
    -------
    gf_tau0 : (L, N, N) np.ndarray
        The unequal time Green's functions for the times 'τ=0, Δτ, ..., (L-1) Δτ'

    Notes
    -----
    The unequal-time Green's function :math:'G(τ_1, τ_2)' for 'τ_1 > τ_2'
    with 'τ = l Δτ'is defined as:
    .. math::
        G(τ_1, τ_2) = B_{l_1 - 1} ... B_{l_2} G_{l_2}

    where 'G_l' is the equal time Green's function.
    For 'τ_2 = 0' and 'τ = τ_1 > 0' this results in
    .. math::
        G(τ=l Δτ, 0) = B_{l-1} ... B_0 G_0
    """
    num_times, num_sites, _ = bmats.shape
    gf = np.zeros((num_times, num_sites, num_sites), dtype=np.float64)
    gf[0] = gf0
    for t in range(1, num_times):
        indices = np.arange(t - 1, -1, -1)
        bmat_seq = bmats[indices]
        bmat_prod = mdot(bmat_seq)
        gf[t] = np.dot(bmat_prod, gf0)
    return gf


# =========================================================================
# Rank-1 update Monte carlo methods
# =========================================================================


@njit(void(expk_t, float64, conf_t, bmat_t, bmat_t, int64, int64), **jkwargs)
def update(expk, nu, config, bmats_up, bmats_dn, i, t):
    r"""Updates the configuration and the corresponding time-step matrices.

    Parameters
    ----------
    expk : np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'
    config : (N, L) np.ndarray
        The configuration or Hubbard-Stratonovich field.
    bmats_up : (L, N, N) np.ndarray
        The spin-up time step matrices.
    bmats_dn : (L, N, N) np.ndarray
        The spin-down time step matrices.
    i : int
        The site index :math:'i' of the proposed spin-flip.
    t : int
        The time-step index :math:'t' of the proposed spin-flip.
    """
    config[i, t] = -config[i, t]
    update_timestep_mats(expk, nu, config, bmats_up, bmats_dn, t)


@njit(nt.UniTuple(float64, 2)(float64, conf_t, gmat_t, gmat_t, int64, int64), **jkwargs)
def compute_determinants(nu, config, gf_up, gf_dn, i, t):
    r"""Computes the determinants used for computing the Metropolis acceptance ratio.

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
    d_up : float
        The spin up determinant.
    d_dn : float
        The spin down determinant.

    Notes
    -----
    The determiants are computed via the Green's function:
    ..math::
        α_σ = e^{-2 σ ν s(i, t)} - 1
        d_σ = 1 + (1 - G_{ii, σ}) α_σ
    """
    arg = -2 * nu * config[i, t]
    alpha_up = np.expm1(UP * arg)
    alpha_dn = np.expm1(DN * arg)
    d_up = 1 + alpha_up * (1 - gf_up[i, i])
    d_dn = 1 + alpha_dn * (1 - gf_dn[i, i])
    return d_up, d_dn


@njit(nt.float64(float64, conf_t, gmat_t, gmat_t, int64, int64), **jkwargs)
def compute_acceptance_fast(nu, config, gf_up, gf_dn, i, t):
    r"""Computes the Metropolis acceptance via the fast update scheme.

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

    Notes
    -----
    A spin-flip of site i and time t is accepted, if d<r:
    ..math::
        α_σ = e^{-2 σ ν s(i, t)} - 1
        d_σ = 1 + (1 - G_{ii, σ}) α_σ
        d = d_↑ d_↓
    """
    arg = -2 * nu * config[i, t]
    alpha_up = np.expm1(UP * arg)
    alpha_dn = np.expm1(DN * arg)
    d_up = 1 + alpha_up * (1 - gf_up[i, i])
    d_dn = 1 + alpha_dn * (1 - gf_dn[i, i])
    return min(abs(d_up * d_dn), 1.0)


@njit(void(float64, conf_t, gmat_t, gmat_t, int64, int64), **jkwargs)
def update_greens(nu, config, gf_up, gf_dn, i, t):
    r"""Performs a Sherman-Morrison update of the Green's function.

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

    Notes
    -----
    The update of the Green's function *before* flipping spin at site i and time t
    is defined as
    ..math::
        G_σ = G_σ - (α_σ / d_σ) u_σ w_σ^T
        u_σ = [I - G_σ] e_i
        w_σ = G_σ^T e_i
        α_σ = e^{-2 σ ν s(i, t)} - 1
        d_σ = 1 + (1 - G_{ii, σ}) α_σ

    References
    ----------
    .. [1] Z. Bai et al., “Numerical Methods for Quantum Monte Carlo Simulations
           of the Hubbard Model”, in Series in Contemporary Applied Mathematics,
           Vol. 12 (June 2009), p. 1.
    """
    # Compute alphas
    arg = -2 * nu * config[i, t]
    alpha_up = np.expm1(UP * arg)
    alpha_dn = np.expm1(DN * arg)
    # Compute acceptance ratios
    d_up = 1 + alpha_up * (1 - gf_up[i, i])
    d_dn = 1 + alpha_dn * (1 - gf_dn[i, i])
    # Temporary array vor (G-I) e_i
    tmp = np.zeros((config.shape[0], 1), dtype=np.float64)
    # Update spin up Green's function
    tmp[:, 0] = gf_up[:, i]
    tmp[i, 0] -= 1
    gf_up += (alpha_up / d_up) * tmp * gf_up[i, :]
    # Update spin down Green's function
    tmp[:, 0] = gf_dn[:, i]
    tmp[i, 0] -= 1
    gf_dn += (alpha_dn / d_dn) * tmp * gf_dn[i, :]


@njit(void(float64, conf_t, gmat_t, gmat_t, int64, int64), **jkwargs)
def update_greens_blas(nu, config, gf_up, gf_dn, i, t):
    r"""Performs a Sherman-Morrison update of the Green's function.

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

    Notes
    -----
    The update of the Green's function *before* flipping spin at site i and time t
    is defined as
    ..math::
        G_σ = G_σ - (α_σ / d_σ) u_σ w_σ^T
        u_σ = [I - G_σ] e_i
        w_σ = G_σ^T e_i
        α_σ = e^{-2 σ ν s(i, t)} - 1
        d_σ = 1 + (1 - G_{ii, σ}) α_σ

    References
    ----------
    .. [1] Z. Bai et al., “Numerical Methods for Quantum Monte Carlo Simulations
           of the Hubbard Model”, in Series in Contemporary Applied Mathematics,
           Vol. 12 (June 2009), p. 1.
    """
    # Compute alphas and determinants
    arg = -2 * nu * config[i, t]
    alpha_up = np.expm1(UP * arg)
    alpha_dn = np.expm1(DN * arg)
    # Temporary array vor (G-I) e_i
    tmp = np.zeros(config.shape[0], dtype=np.float64)
    # Update spin up Green's function
    tmp[:] = gf_up[:, i]
    tmp[i] -= 1.0
    dger(alpha_up / (1.0 - alpha_up * tmp[i]), tmp, gf_up[i, :], gf_up)
    # Update spin down Green's function
    tmp[:] = gf_dn[:, i]
    tmp[i] -= 1.0
    dger(alpha_dn / (1.0 - alpha_dn * tmp[i]), tmp, gf_dn[i, :], gf_dn)


@njit(void(bmat_t, bmat_t, gmat_t, gmat_t, int64), **jkwargs)
def wrap_up_greens(bmats_up, bmats_dn, gf_up, gf_dn, t):
    r"""Wraps the Green's functions from the time step :math:'t' up to :math:'t+1'.

    This method has to be called after a time-step in order to prepare the
    Green's function for the next higher time slice.

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

    Notes
    -----
    Wrapping the Green's function from the time step :math:'t' up to :math:'t+1'
    is defined as
    ..math::
        G_σ(t+1) = B_σ(t) G_σ(t) B_σ^{-1}(t)

    References
    ----------
    .. [1] Z. Bai et al., “Numerical Methods for Quantum Monte Carlo Simulations
           of the Hubbard Model”, in Series in Contemporary Applied Mathematics,
           Vol. 12 (June 2009), p. 1.
    """
    b_up = bmats_up[t]
    b_dn = bmats_dn[t]
    gf_up[:, :] = np.dot(np.dot(b_up, gf_up), la.inv(b_up))
    gf_dn[:, :] = np.dot(np.dot(b_dn, gf_dn), la.inv(b_dn))


@njit(void(bmat_t, bmat_t, gmat_t, gmat_t, int64), **jkwargs)
def wrap_down_greens(bmats_up, bmats_dn, gf_up, gf_dn, t):
    r"""Wraps the Green's functions from the time step :math:'t' down to :math:'t-1'.

    This method has to be called after a time-step in order to prepare the
    Green's function for the next lower time slice.

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

    Notes
    -----
    Wrapping the Green's function from the time step :math:'t' up to :math:'t-1'
    is defined as
    ..math::
        G_σ(t-1) = B_σ(t-1) G_σ(t-1) B_σ^{-1}(t-1)

    References
    ----------
    .. [1] Z. Bai et al., “Numerical Methods for Quantum Monte Carlo Simulations
           of the Hubbard Model”, in Series in Contemporary Applied Mathematics,
           Vol. 12 (June 2009), p. 1.
    """
    idx = t - 1
    b_up = bmats_up[idx]
    b_dn = bmats_dn[idx]
    gf_up[:, :] = np.dot(np.dot(la.inv(b_up), gf_up), b_up)
    gf_dn[:, :] = np.dot(np.dot(la.inv(b_dn), gf_dn), b_dn)


@njit(int64(expk_t, float64, conf_t, bmat_t, bmat_t, gmat_t, gmat_t,
            int64[::1], float64[:], int64[::1], int64), **jkwargs)
def dqmc_time_step(expk, nu, config, bmats_up, bmats_dn, gf_up, gf_dn, sgndet, logdet,
                   sites, t):
    """Accelerated inner loop of the DQMC iteration."""
    # Iterate over all lattice sites randomly
    accepted = 0
    np.random.shuffle(sites)
    for i in sites:
        # Propose spin-flip in configuration
        arg = -2 * nu * config[i, t]
        alpha_up = np.expm1(UP * arg)
        alpha_dn = np.expm1(DN * arg)
        d_up = 1 + (1 - gf_up[i, i]) * alpha_up
        d_dn = 1 + (1 - gf_dn[i, i]) * alpha_dn
        # Check if move is accepted
        if np.random.random() < abs(d_up * d_dn):
            # Move accepted
            accepted += 1
            # Update Green's functions *before* updating configuration
            update_greens(nu, config, gf_up, gf_dn, i, t)
            # Update signs
            if d_up < 0:
                sgndet[0] = -sgndet[0]
            if d_dn < 0:
                sgndet[1] = -sgndet[1]
            # Update determinants
            logdet[0] -= np.log(np.abs(d_up))
            logdet[1] -= np.log(np.abs(d_dn))
            # Actually update configuration and B-matrices *after* GF update
            config[i, t] = -config[i, t]
            # update_timestep_mats(expk, nu, config, bmats_up, bmats_dn, t)

    # Update time-step matrix of the current time slice before next time slice.
    # Can be done outside the inner loop over the lattice sites since it only uses
    # the i-th spin and i-th row/column of the Green's functions.
    update_timestep_mats(expk, nu, config, bmats_up, bmats_dn, t)

    return accepted


def dqmc_iteration(expk, nu, config, bmats_up, bmats_dn, gf_up, gf_dn, sgndet, logdet,
                   nwraps, nprod):
    r"""Runs one iteration of the rank-1 DQMC-scheme.

    Parameters
    ----------
    expk : np.ndarray
        The matrix exponential of the kinetic hamiltonian.
    nu : float
        The parameter ν defined by :math:'\cosh(ν) = e^{U Δτ / 2}'.
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
    sgndet : (2, ) np.ndarray
        The signs of the determinants of the Green's functions.
    logdet : (2, ) np.ndarray
        The logarithm of the determinants of the Green's functions.
    nwraps : int, optional
        Number of time slices after which the Green's functions are recomputed.
        Must be a multiple of the number of imaginary time steps!  If a `0` is passed
        the Green's function is not recomputed between time steps.
    nprod : int, optional
        The number of direct products used in the matrix product stabilization.
        Must be a multiple of the number of imaginary time steps! If a `0` is passed
        no stabilization is used.

    Returns
    -------
    accepted : int
        The number of accepted spin flips.
    """
    accepted = 0
    sites = np.arange(config.shape[0], dtype=np.int64)
    # Iterate over all time-steps
    for t in range(config.shape[1]):
        # Iterate over all lattice sites and perform updates
        accepted += dqmc_time_step(expk, nu, config, bmats_up, bmats_dn,
                                   gf_up, gf_dn, sgndet, logdet, sites, t)

        # Recompute Green's function for next slice `t+1` after several time slices,
        # otherwise wrap Green's functions up to next time slice.
        tp1 = t + 1
        if nwraps and tp1 % nwraps == 0:
            if nprod > 0:
                compute_greens_qrd(bmats_up, bmats_dn, gf_up, gf_dn, sgndet, logdet,
                                   tp1, nprod)
            else:
                compute_greens(bmats_up, bmats_dn, gf_up, gf_dn, sgndet, logdet, tp1)
        else:
            wrap_up_greens(bmats_up, bmats_dn, gf_up, gf_dn, t)

    return accepted


@njit(
    (gmat_t, gmat_t, int64[::1], gmat_t, gmat_t, float64[:], float64[:],
     float64[:], float64[:]),
    **jkwargs
)
def accumulate_measurements(gf_up, gf_dn, sgns, gfout_up, gfout_dn, n_up, n_dn,
                            n_dbl, moment):
    """Acummulate (unnormalized) equal time measurements of observables.

    Parameters
    ----------
    gf_up : (N, N) np.ndarray
        The current spin-up Green's function matrix :math:`G_↑`.
    gf_dn : (N, N) np.ndarray
        The current spin-down Green's function matrix :math:`G_↓`.
    sgns : (2,) np.ndarray
        The sign of the determinant of the Green's function of both spins.
    gfout_up : (N, N) np.ndarray
        The output spin-up Green's function matrix :math:`G_↑`.
    gfout_dn : (N, N) np.ndarray
        The output spin-down Green's function matrix :math:`G_↓`.
    n_up : (N,) np.ndarray
        The output array of the spin-up occupation :math:`<n_↑>`.
    n_dn : (N,) np.ndarray
        The output array of the spin-down occupation :math:`<n_↓>`.
    n_dbl : (N,) np.ndarray
        The output array of the double occupation :math:`<n_↑ n_↓>`.
    moment : (N,) np.ndarray
        The output array of the local moment :math:`<m_z^2>`.

    Notes
    -----
    The measured observables are defined as
    ..math::
        <n_{iσ}>  = 1 - [G_σ]_{ii}
        <n_↑ n_↓> = (1 - [G_↑]_{ii}) (1 - [G_↓]_{ii})
        <m_z^2> = <n_↑> + <n_↓> - 2 <n_↑ n_↓>
    """
    signfac = sgns[0] * sgns[1]

    _n_up = 1 - np.diag(gf_up)
    _n_dn = 1 - np.diag(gf_dn)
    _n_double = _n_up * _n_dn
    _moment = _n_up + _n_dn - 2 * _n_double

    gfout_up += gf_up * signfac
    gfout_dn += gf_dn * signfac
    n_up += _n_up * signfac
    n_dn += _n_dn * signfac
    n_dbl += _n_double * signfac
    moment += _moment * signfac
