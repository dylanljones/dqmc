# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import random
import logging
import math
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from numba import njit
from dqmc import HubbardModel, UP, DN, init_configuration
from dqmc.utils import Plot

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
    bmats_up[t] = exp_k * np.exp(UP * nu * config[:, t])
    bmats_dn[t] = exp_k * np.exp(DN * nu * config[:, t])


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


class DQMC:

    def __init__(self, model, num_timesteps):
        # Compute and check time step size
        dtau = model.beta / num_timesteps
        check = model.u * model.hop * dtau ** 2
        if check > 0.1:
            logger.warning(f"Increase number of time steps: Check-value {check} should be <0.1!")

        # Compute factor and matrix exponential of kinetic hamiltonian
        ham_k = model.hamiltonian_kinetic(periodic=True)
        self.exp_k = la.expm(dtau * ham_k)
        self.nu = math.acosh(math.exp(model.u * dtau / 2.))

        # Initialize configuration
        self.config = init_configuration(model.num_sites, num_timesteps)

        # Pre-compute time flow matrices
        self.bmats_up = None
        self.bmats_dn = None
        self.compute_time_step_matrices()

        # Initialize Greens functions
        self.gf_up = None
        self.gf_dn = None
        self.compute_greens_functions()

        # Analysis
        self.logdets_up = list()
        self.logdets_dn = list()

    def compute_time_step_matrices(self):
        self.bmats_up = compute_time_step_matrices(self.exp_k, self.nu, self.config, sigma=UP)
        self.bmats_dn = compute_time_step_matrices(self.exp_k, self.nu, self.config, sigma=DN)

    def update_time_step_matrices(self, t):
        update_time_step_matrices(self.exp_k, self.nu, self.config, self.bmats_up, self.bmats_dn, t)

    def compute_greens_functions(self):
        eye = np.eye(self.bmats_up[0].shape[0])
        time_order = np.arange(self.config.shape[1])[::1]
        self.gf_up = la.inv(eye + compute_time_flow_map(self.bmats_up, time_order))
        self.gf_dn = la.inv(eye + compute_time_flow_map(self.bmats_dn, time_order))

    def update_greens(self, i, t):
        # Compute alphas
        arg = -2 * self.nu * self.config[i, t]
        alpha_up = np.expm1(UP * arg)
        alpha_dn = np.expm1(DN * arg)
        # Compute c-vectors for all j
        c_up = -alpha_up * self.gf_up[i, :]
        c_dn = -alpha_dn * self.gf_dn[i, :]
        # Add diagonal elements where j=i
        c_up[i] += alpha_up
        c_dn[i] += alpha_dn
        # Compute b-vectors for all k
        b_up = self.gf_up[:, i] / (1 + c_up[i])
        b_dn = self.gf_dn[:, i] / (1 + c_dn[i])
        # Compute outer product of b and c and update GF for all j and k
        self.gf_up += -np.dot(b_up[:, None], c_up[None, :])
        self.gf_dn += -np.dot(b_dn[:, None], c_dn[None, :])

    def wrap_greens(self, t):
        b_up = self.bmats_up[t]
        b_dn = self.bmats_dn[t]
        self.gf_up = np.dot(np.dot(b_up, self.gf_up), la.inv(b_up))
        self.gf_dn = np.dot(np.dot(b_dn, self.gf_dn), la.inv(b_dn))

    def dqmc_iteration(self):
        # Iterate over all time-steps
        accepted = 0
        for t in range(self.config.shape[1]):
            # Wrap the green's function for the current time slice
            self.wrap_greens(t)
            # Iterate over all lattice sites
            for i in range(self.config.shape[0]):

                # Compute acceptance ratio
                arg = -2 * self.nu * self.config[i, t]
                d_up = 1 + (1 - self.gf_up[i, i]) * np.expm1(UP * arg)
                d_dn = 1 + (1 - self.gf_dn[i, i]) * np.expm1(DN * arg)
                self.logdets_up.append(math.log(abs(d_up)))
                self.logdets_up.append(math.log(abs(d_up)))
                d = min(abs(d_up * d_dn), 1.0)

                # Check if move is accepted
                accept = random.random() < d
                if accept:
                    accepted += 1
                    # Move accepted: update configuration and Green's functions
                    self.update_greens(i, t)
                    self.config[i, t] *= -1
                    self.update_time_step_matrices(t)

                logger.debug("(%s, %s) d=%.2f accept: %s", i, t, d, accept)

        return accepted / self.config.size

    def simulate(self, sweeps):
        ratios = list()
        for sweep in range(sweeps):
            acc_ratio = self.dqmc_iteration()
            ratios.append(acc_ratio)
            logger.info("[%3d] Ratio: %.2f", sweep, acc_ratio)
            self.compute_time_step_matrices()
            self.compute_greens_functions()
        return ratios


def main():
    model = HubbardModel(10, u=2.0, hop=1.0, beta=2)
    dqmc = DQMC(model, num_timesteps=50)

    ratios = dqmc.simulate(1000)

    # plot acceptance ratio
    fig, ax = plt.subplots()
    ax.plot(ratios)
    ax.grid()
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlim(0, None)
    ax.set_xlabel("Seeep")
    ax.set_ylabel("Ratio")

    # fig, ax = plt.subplots()
    # ax.plot(dqmc.logdets_up)
    # ax.plot(dqmc.logdets_dn)
    plt.show()


if __name__ == "__main__":
    main()
