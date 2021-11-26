# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import math
import random
import logging
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from numba import njit
from dqmc import HubbardModel, UP, DN

logger = logging.getLogger("dqmc")
logger.setLevel(logging.INFO)

random.seed(0)


@njit
def compute_time_step_matrix(exp_k, nu, config, t, sigma):
    r"""Computes the time step matrix :math:'B_σ(h_t)'.

    The time step matrix :math:'B_σ(h_t)' is defined as
    ..math::
        B_σ(h_t) = e^k e^{σ ν V_t(h_t)}
    """
    return exp_k * np.exp(sigma * nu * config[:, t])


@njit
def compute_time_step_matrices(exp_k, nu, config):
    r"""Computes the time step matrices :math:'B_σ(h_t)' for all times `t` and both spins."""
    num_sites, num_timesteps = config.shape
    bmats_up = np.zeros((num_timesteps, num_sites, num_sites), dtype=np.float64)
    bmats_dn = np.zeros((num_timesteps, num_sites, num_sites), dtype=np.float64)
    for t in range(num_timesteps):
        bmats_up[t] = compute_time_step_matrix(exp_k, nu, config, t, sigma=UP)
        bmats_dn[t] = compute_time_step_matrix(exp_k, nu, config, t, sigma=DN)
    return bmats_up, bmats_dn


@njit
def update_time_step_matrices(exp_k, nu, config, bmats_up, bmats_dn, t):
    r"""Updates one time step matrices :math:'B_σ(h_t)' for both spins and one time step."""
    bmats_up[t] = compute_time_step_matrix(exp_k, nu, config, t, sigma=UP)
    bmats_dn[t] = compute_time_step_matrix(exp_k, nu, config, t, sigma=DN)


@njit
def compute_time_flow_map(bmats, order=None):
    r"""Computes the fermion time flow map matrix :math:'A_σ(h)'.

    The matrix :math:'A_σ' is defined as
    ..math::
        A_σ = B_σ(1) B_σ(2) ... B_σ(L)
    """
    if order is None:
        order = np.arange(len(bmats))
    # First matrix
    b_prod = bmats[order[0]]
    # Following matrices multiplied with dot-product
    for i in order[1:]:
        b_prod = np.dot(b_prod, bmats[i])
    return b_prod


@njit
def compute_m_matrices(bmats_up, bmats_dn, order=None):
    r"""Computes the matrix :math:'I + A_σ(h)' for both spins."""
    if order is None:
        order = np.arange(len(bmats_up))
    eye = np.eye(bmats_up[0].shape[0])
    m_up = eye + compute_time_flow_map(bmats_up, order)
    m_dn = eye + compute_time_flow_map(bmats_dn, order)
    return m_up, m_dn


def measure_occupation(m_up, m_dn):
    occ_up = 1 - np.diag(la.inv(m_up))
    occ_dn = 1 - np.diag(la.inv(m_dn))
    return occ_up, occ_dn


def main():
    time_dir = -1
    warmup = 500
    measurement = 1000
    sweeps = warmup + measurement

    num_timesteps = 50
    model = HubbardModel(num_sites=10, u=10, hop=.1, eps=-5, beta=2)

    # Initialize QMC simulation
    # -------------------------

    ham_k = model.hamiltonian_kinetic(periodic=True)
    print(ham_k)

    # Compute and check time step size
    dtau = model.beta / num_timesteps
    check = model.u * model.hop * dtau ** 2
    if check > 0.1:
        logger.warning(f"Increase number of time steps: Check-value {check} should be <0.1!")

    # Initialize configuration
    config = np.random.choice([-1, +1], size=(model.num_sites, num_timesteps))

    # Compute factor and matrix exponential of kinetic hamiltonian
    nu = math.acosh(math.exp(model.u * dtau / 2.))

    exp_k = la.expm(dtau * ham_k)

    # Run QMC warmup sweeps
    # ---------------------

    # Pre-compute time flow matrices
    bmats_up, bmats_dn = compute_time_step_matrices(exp_k, nu, config)

    # Compute M-matrices
    time_order = np.arange(config.shape[1])[::time_dir]
    m_up, m_dn = compute_m_matrices(bmats_up, bmats_dn, time_order)

    # Initialize the determinant product
    old_det = la.det(m_up) * la.det(m_dn)

    # Warmup-sweeps
    ratios = list()
    times = np.arange(config.shape[1])[::time_dir]
    sites = np.arange(config.shape[0])

    occupation_up, occupation_dn = 0, 0
    for sweep in range(sweeps):
        logger.debug("SWEEP %s", sweep)
        logger.debug("----------------")
        accepted = 0

        # Iterate over all time-steps
        for t in times:
            # time_order = np.arange(config.shape[1])

            # Iterate over all lattice sites
            np.random.shuffle(sites)
            for i in sites:
                # Propose update by flipping spin in confguration
                config[i, t] *= -1
                update_time_step_matrices(exp_k, nu, config, bmats_up, bmats_dn, t)
                # bmats_up, bmats_dn = compute_time_step_matrices(exp_k, nu, config)

                # Compute determinant product of the new configuration
                m_up, m_dn = compute_m_matrices(bmats_up, bmats_dn, time_order)
                det_up = la.det(m_up)
                det_dn = la.det(m_dn)
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
                    update_time_step_matrices(exp_k, nu, config, bmats_up, bmats_dn, t)
                    # bmats_up, bmats_dn = compute_time_step_matrices(exp_k, nu, config)

                logger.debug("(%s, %s) det_up=%.2e det_dn=%.2e d=%.2f accept: %s",
                             i, t, det_up, det_dn, d, accept)

        # Compute and save acceptance ratio
        acc_ratio = accepted / config.size
        ratios.append(acc_ratio)
        logger.info("[%3d] Ratio: %.2f", sweep, acc_ratio)

        # perform measurements
        if sweep > warmup:
            m_up, m_dn = compute_m_matrices(bmats_up, bmats_dn, time_order)
            occ_up, occ_dn = measure_occupation(m_up, m_dn)
            occupation_up += occ_up
            occupation_dn += occ_dn

    # Normalize measurements
    occupation_up /= measurement
    occupation_dn /= measurement
    print(occupation_up, occupation_dn)
    # plot acceptance ratio
    fig, ax = plt.subplots()
    ax.plot(ratios)
    ax.grid()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, None)
    ax.set_xlabel("Seeep")
    ax.set_ylabel("Ratio")
    plt.show()


if __name__ == "__main__":
    main()
