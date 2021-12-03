# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import random
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from hypothesis import example, given, strategies as st
from dqmc import dqmc, hubbard_hypercube

DECIMALS = 5


def _init_1d(num_sites, num_timesteps, u, eps=0.0, hop=1.0, mu=0.0, beta=0.0, periodic=True):
    model = hubbard_hypercube(num_sites, u, eps, hop, mu, beta, periodic)
    exp_k, nu, config = dqmc.init_qmc(model, num_timesteps)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(exp_k, nu, config)
    return exp_k, nu, config, bmats_up, bmats_dn


@given(st.floats(0.1, 10), st.floats(0.1, 10.), st.integers(0, 4))
def test_update_greens_function(u, beta, i):
    num_sites = 5
    t = 0  # Only use first time slice to prevent need for wrapping

    exp_k, nu, config, bmats_up, bmats_dn = _init_1d(num_sites, 100, u=u, beta=beta)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(exp_k, nu, config)
    bmat_order = np.arange(config.shape[1], dtype=np.int64)[::-1]
    gf_up, gf_dn = dqmc.compute_greens(bmats_up, bmats_dn, bmat_order)

    # -----------Update Green's function --------------
    dqmc.update_greens(nu, config, gf_up, gf_dn, i, t)
    dqmc.update(exp_k, nu, config, bmats_up, bmats_dn, i, t)
    gf_up_ref, gf_dn_ref = dqmc.compute_greens(bmats_up, bmats_dn, bmat_order)
    # -------------------------------------------------

    assert_array_almost_equal(gf_up, gf_up_ref, decimal=DECIMALS)
    assert_array_almost_equal(gf_dn, gf_dn_ref, decimal=DECIMALS)


@given(st.floats(0.1, 10), st.floats(0.1, 10.), st.integers(0, 4))
def test_update_greens_function2(u, beta, i):
    num_sites = 5
    t = 0  # Only use first time slice to prevent need for wrapping

    exp_k, nu, config, bmats_up, bmats_dn = _init_1d(num_sites, 100, u=u, beta=beta)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(exp_k, nu, config)
    bmat_order = np.arange(config.shape[1], dtype=np.int64)[::-1]
    gf_up, gf_dn = dqmc.compute_greens(bmats_up, bmats_dn, bmat_order)

    # -----------Update Green's function --------------
    dqmc.update_greens2(nu, config, gf_up, gf_dn, i, t)
    dqmc.update(exp_k, nu, config, bmats_up, bmats_dn, i, t)
    gf_up_ref, gf_dn_ref = dqmc.compute_greens(bmats_up, bmats_dn, bmat_order)
    # -------------------------------------------------

    assert_array_almost_equal(gf_up, gf_up_ref, decimal=DECIMALS)
    assert_array_almost_equal(gf_dn, gf_dn_ref, decimal=DECIMALS)


@given(st.floats(0.1, 10), st.floats(0.1, 10.), st.integers(0, 4))
def test_compare_update_greens_function(u, beta, i):
    num_sites = 5
    t = 0  # Only use first time slice to prevent need for wrapping

    exp_k, nu, config, bmats_up, bmats_dn = _init_1d(num_sites, 100, u=u, beta=beta)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(exp_k, nu, config)
    bmat_order = np.arange(config.shape[1], dtype=np.int64)[::-1]
    gf_up, gf_dn = dqmc.compute_greens(bmats_up, bmats_dn, bmat_order)
    gf_up2, gf_dn2 = np.copy(gf_up), np.copy(gf_dn)

    # -----------Update Green's functions --------------
    dqmc.update_greens(nu, config, gf_up, gf_dn, i, t)
    dqmc.update_greens2(nu, config, gf_up2, gf_dn2, i, t)
    # -------------------------------------------------

    assert_array_almost_equal(gf_up, gf_up2, decimal=DECIMALS)
    assert_array_almost_equal(gf_dn, gf_dn2, decimal=DECIMALS)
