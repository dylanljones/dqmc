# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import random
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from numba import njit
from dqmc import HubbardModel, UP, DN

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
def compute_time_step_matrices(exp_k, nu, config, sigma):
    r"""Computes the time step matrices :math:'B_σ(h_t)' for all times `t`."""
    num_sites, num_timesteps = config.shape
    bmats = np.zeros((num_timesteps, num_sites, num_sites))
    for t in range(num_timesteps):
        bmats[t] = exp_k * np.exp(sigma * nu * config[:, t])
    return bmats


@njit
def update_time_step_matrices(exp_k, nu, config, bmats_up, bmats_dn, t):
    r"""Updates one time step matrix :math:'B_σ(h_t)' for one time step."""
    bmats_up[t] = compute_time_step_matrix(exp_k, nu, config, t, sigma=UP)
    bmats_dn[t] = compute_time_step_matrix(exp_k, nu, config, t, sigma=DN)


@njit
def compute_time_flow_map(bmats):
    r"""Computes the fermion time flow map matrix :math:'A_σ(h)'.

    The matrix :math:'A_σ' is defined as
    ..math::
        A_σ = B_σ(1) B_σ(2) ... B_σ(L)
    """
    # First matrix
    b_prod = bmats[0]
    # Following matrices multiplied with dot-product
    for i in range(1, bmats.shape[0]):
        b_prod = np.dot(b_prod, bmats[i])
    return b_prod


def warmup_det(config, nu, exp_k, sweeps=200):
    # Pre-compute time flow matrices
    bmats_up = compute_time_step_matrices(exp_k, nu, config, sigma=UP)
    bmats_dn = compute_time_step_matrices(exp_k, nu, config, sigma=DN)

    # Initialize the determinant product
    eye = np.eye(bmats_up[0].shape[0])
    m_up = eye + compute_time_flow_map(bmats_up)
    m_dn = eye + compute_time_flow_map(bmats_dn)
    old_det = la.det(m_up) * la.det(m_dn)

    # Warmup-sweeps
    ratios = list()
    for _ in range(sweeps):
        accepted = 0
        # Iterate over all time-steps
        for t in range(config.shape[1]):
            # Iterate over all lattice sites
            for i in range(config.shape[0]):
                # Propose update by flipping spin in confguration
                config[i, t] = -config[i, t]
                update_time_step_matrices(exp_k, nu, config, bmats_up, bmats_dn, t)

                # Compute determinant product of the new configuration
                eye = np.eye(bmats_up[0].shape[0])
                m_up = eye + compute_time_flow_map(bmats_up)
                m_dn = eye + compute_time_flow_map(bmats_dn)
                new_det = la.det(m_up) * la.det(m_dn)

                # Compute acceptance ratio
                d = abs(new_det / old_det)

                # Check if move is accepted
                if random.random() < d:
                    # Move accepted: Continue using the new configuration
                    accepted += 1
                    old_det = new_det
                else:
                    # Move not accepted: Revert to the old configuration by updating again
                    config[i, t] = -config[i, t]
                    update_time_step_matrices(exp_k, nu, config, bmats_up, bmats_dn, t)

        # Compute and save acceptance ratio
        acc_ratio = accepted / config.size
        ratios.append(acc_ratio)
        print(acc_ratio)

    # plot acceptance ratio
    fig, ax = plt.subplots()
    ax.plot(ratios)
    plt.show()


def main():
    model = HubbardModel(10, u=2.0, hop=1.0, beta=4)
    num_timesteps = 100

    # Compute and check time step size
    dtau = model.beta / num_timesteps
    check = model.u * model.hop * dtau ** 2
    if check > 0.1:
        print(f"Increase number of time steps: Check-value {check} should be <0.1!")

    # Initialize configuration
    config = 2 * np.random.randint(0, 2, size=(model.num_sites, num_timesteps), dtype=np.int8) - 1

    # Compute factor and matrix exponential of kinetic hamiltonian
    nu = np.arccosh(np.exp(model.u * dtau / 2.)) if model.u else 0
    exp_k = la.expm(dtau * model.hamiltonian_kinetic())

    warmup_det(config, nu, exp_k, sweeps=1000)


if __name__ == "__main__":
    main()
