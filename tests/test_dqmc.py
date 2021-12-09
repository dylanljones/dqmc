# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import random
import numpy as np
import scipy.linalg as la
from functools import reduce
from numpy.testing import assert_array_equal, assert_array_almost_equal
from hypothesis import given, strategies as st
from dqmc import dqmc, hubbard_hypercube


def test_init_qmc_atomic():
    model = hubbard_hypercube(5, u=1, hop=0., beta=1.0)
    exp_k, nu, config = dqmc.init_qmc(model, 100)
    assert_array_equal(np.eye(model.num_sites), exp_k)
    assert abs(nu - 0.1) < 1e-3


def test_init_qmc_noninter():
    model = hubbard_hypercube(5, u=0, hop=1., beta=1.0)
    exp_k, nu, config = dqmc.init_qmc(model, 100)

    expected = np.eye(model.num_sites)
    np.fill_diagonal(expected[1:, :], -0.01)
    np.fill_diagonal(expected[:, 1:], -0.01)
    assert_array_almost_equal(expected, exp_k, decimal=3)
    assert nu == 0


def test_init_qmc_zerot():
    model = hubbard_hypercube(5, u=1, hop=1., beta=0.0)
    exp_k, nu, config = dqmc.init_qmc(model, 100)
    assert_array_equal(np.eye(model.num_sites), exp_k)
    assert nu == 0


@given(st.floats(0.1, 5), st.floats(0, 5), st.floats(0.1, 10))
def test_compute_timestep_mat(u, mu, beta):
    model = hubbard_hypercube(5, u, mu=mu, beta=beta)
    exp_k, nu, config = dqmc.init_qmc(model, 200)

    t = random.randint(0, 199)

    # Spin up
    sigma = +1
    diag = sigma * nu * config[:, t]
    expected = np.dot(exp_k, la.expm(np.diag(diag)))
    result = dqmc.compute_timestep_mat(exp_k, nu, config, t, sigma)
    assert_array_almost_equal(expected, result, decimal=12)

    # Spin down
    sigma = -1
    diag = sigma * nu * config[:, t]
    expected = np.dot(exp_k, la.expm(np.diag(diag)))
    result = dqmc.compute_timestep_mat(exp_k, nu, config, t, sigma)
    assert_array_almost_equal(expected, result, decimal=12)


@given(st.floats(0.1, 5), st.floats(0, 5), st.floats(0.1, 10))
def test_compute_timestep_mats(u, mu, beta):
    model = hubbard_hypercube(5, u, mu=mu, beta=beta)
    exp_k, nu, config = dqmc.init_qmc(model, 200)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(exp_k, nu, config)

    # Check order of B-matrices
    first_up = dqmc.compute_timestep_mat(exp_k, nu, config, t=0, sigma=+1)
    first_dn = dqmc.compute_timestep_mat(exp_k, nu, config, t=0, sigma=-1)
    last_up = dqmc.compute_timestep_mat(exp_k, nu, config, t=199, sigma=+1)
    last_dn = dqmc.compute_timestep_mat(exp_k, nu, config, t=199, sigma=-1)
    assert_array_equal(last_up, bmats_up[-1])
    assert_array_equal(last_dn, bmats_dn[-1])
    assert_array_equal(first_up, bmats_up[0])
    assert_array_equal(first_dn, bmats_dn[0])


@given(st.floats(0.1, 5), st.floats(0, 5), st.floats(0.1, 10))
def test_update_timestep_mats(u, mu, beta):
    model = hubbard_hypercube(5, u, mu=mu, beta=beta)
    exp_k, nu, config = dqmc.init_qmc(model, 200)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(exp_k, nu, config)

    t = random.randint(0, 199)
    i = random.randint(0, 4)

    config[i, t] = -config[i, t]

    bmat_up = dqmc.compute_timestep_mat(exp_k, nu, config, t, +1)
    bmat_dn = dqmc.compute_timestep_mat(exp_k, nu, config, t, -1)
    dqmc.update_timestep_mats(exp_k, nu, config, bmats_up, bmats_dn, t)

    assert_array_equal(bmat_up, bmats_up[t])
    assert_array_equal(bmat_dn, bmats_dn[t])


@given(st.floats(0.1, 5), st.floats(0, 5), st.floats(0.1, 10))
def test_compute_timeflow_map(u, mu, beta):
    model = hubbard_hypercube(5, u, mu=mu, beta=beta)
    exp_k, nu, config = dqmc.init_qmc(model, 200)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(exp_k, nu, config)
    order = np.arange(config.shape[1]).astype(np.int64)

    # Spin up
    expected = reduce(np.dot, bmats_up[order])
    result = dqmc.compute_timeflow_map(bmats_up, order)
    assert_array_equal(expected, result)
    # Spin down
    expected = reduce(np.dot, bmats_dn[order])
    result = dqmc.compute_timeflow_map(bmats_dn, order)
    assert_array_equal(expected, result)


@given(st.integers(1, 5), st.integers(1, 5), st.integers(0, 9), st.integers(0, 99))
def test_update(u, beta, i, t):
    num_sites = 10
    num_timesteps = 100
    eps, mu, hop = 0.0, 0.0, 1.0

    model = hubbard_hypercube(num_sites, u, eps, hop, mu, beta, periodic=True)
    exp_k, nu, config = dqmc.init_qmc(model, num_timesteps)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(exp_k, nu, config)

    old_spin = config[i, t]
    dqmc.update(exp_k, nu, config, bmats_up, bmats_dn, i, t)

    bmats_up_new, bmats_dn_new = dqmc.compute_timestep_mats(exp_k, nu, config)

    assert config[i, t] == -old_spin
    assert_array_equal(bmats_up_new, bmats_up)
    assert_array_equal(bmats_dn_new, bmats_dn)


def test_compute_acceptance_fast():
    pass


def test_compute_greens():
    pass


@given(st.integers(1, 5), st.integers(1, 5), st.integers(0, 9))
def test_update_greens(u, beta, i):
    num_sites = 10
    num_timesteps = 100
    eps, mu, hop = 0.0, 0.0, 1.0
    t = 0  # Only use first time slice to prevent need for wrapping

    model = hubbard_hypercube(num_sites, u, eps, hop, mu, beta, periodic=True)
    exp_k, nu, config = dqmc.init_qmc(model, num_timesteps)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(exp_k, nu, config)

    bmat_order = np.arange(config.shape[1], dtype=np.int64)[::-1]
    gf_up, gf_dn = dqmc.compute_greens(bmats_up, bmats_dn, bmat_order)

    # -----------Update Green's function --------------
    dqmc.update_greens(nu, config, gf_up, gf_dn, i, t)
    dqmc.update(exp_k, nu, config, bmats_up, bmats_dn, i, t)
    gf_up_ref, gf_dn_ref = dqmc.compute_greens(bmats_up, bmats_dn, bmat_order)
    # -------------------------------------------------

    assert_array_almost_equal(gf_up, gf_up_ref, decimal=5)
    assert_array_almost_equal(gf_dn, gf_dn_ref, decimal=5)


def test_wrap_greens():
    pass


def test_iteration_det():
    pass


def test_iteration_fast():
    pass
