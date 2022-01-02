# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

"""Tests for the main dqmc methods.

The deadline of the hypothesis tests have been increased to prevent `flaky` errors.
"""

import numpy as np
import scipy.linalg as la
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose
from hypothesis import given, settings, assume, strategies as st
from dqmc import dqmc, hubbard_hypercube

settings.register_profile("dqmc", deadline=50000, max_examples=100)
settings.load_profile("dqmc")

st_u = st.floats(0.1, 10)
st_beta = st.floats(0.1, 10)
st_mu = st.floats(0, 5)
st_nsites = st.integers(4, 20)
st_ntimes = st.integers(20, 200)
st_i = st.integers(0, 20)
st_t = st.integers(0, 200)


def _init(num_sites, num_timesteps, u, mu, beta, eps=0.0, periodic=True):
    hop = 1.0
    assume(u * hop * (beta / num_timesteps)**2 < 0.1)
    model = hubbard_hypercube(num_sites, u, eps, hop, mu, beta, periodic=periodic)
    exp_k, nu, config = dqmc.init_qmc(model, num_timesteps, 0)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(exp_k, nu, config)
    return exp_k, nu, config, bmats_up, bmats_dn


def test_init_qmc_atomic():
    model = hubbard_hypercube(5, u=1, hop=0., beta=1.0)
    exp_k, nu, config = dqmc.init_qmc(model, 100, 0)
    assert_array_equal(np.eye(model.num_sites), exp_k)
    assert abs(nu - 0.1) < 1e-3


def test_init_qmc_noninter():
    model = hubbard_hypercube(5, u=0, hop=1., beta=1.0)
    exp_k, nu, config = dqmc.init_qmc(model, 100, 0)

    expected = np.eye(model.num_sites)
    np.fill_diagonal(expected[1:, :], -0.01)
    np.fill_diagonal(expected[:, 1:], -0.01)
    assert_array_almost_equal(expected, exp_k, decimal=3)
    assert nu == 0


def test_init_qmc_zerot():
    model = hubbard_hypercube(5, u=1, hop=1., beta=0.0)
    exp_k, nu, config = dqmc.init_qmc(model, 100, 0)
    assert_array_equal(np.eye(model.num_sites), exp_k)
    assert nu == 0


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_t)
def test_compute_timestep_mat(u, mu, beta, num_sites, num_times, t):
    # num_sites, num_times = 10, 200
    eps, hop = 0.0, 1.0
    assume(t < num_times)
    assume(u * hop * (beta / num_times)**2 < 0.1)
    model = hubbard_hypercube(num_sites, u, eps, hop, mu, beta, periodic=True)
    exp_k, nu, config = dqmc.init_qmc(model, num_times, 0)

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


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes)
def test_compute_timestep_mats(u, mu, beta, num_sites, num_times):
    exp_k, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    # Check order of B-matrices
    t1 = num_times - 1
    first_up = dqmc.compute_timestep_mat(exp_k, nu, config, t=0, sigma=+1)
    first_dn = dqmc.compute_timestep_mat(exp_k, nu, config, t=0, sigma=-1)
    last_up = dqmc.compute_timestep_mat(exp_k, nu, config, t=t1, sigma=+1)
    last_dn = dqmc.compute_timestep_mat(exp_k, nu, config, t=t1, sigma=-1)
    assert_array_equal(last_up, bmats_up[-1])
    assert_array_equal(last_dn, bmats_dn[-1])
    assert_array_equal(first_up, bmats_up[0])
    assert_array_equal(first_dn, bmats_dn[0])


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_i, st_t)
def test_update_timestep_mats(u, mu, beta, num_sites, num_times, i, t):
    assume(i < num_sites)
    assume(t < num_times)
    exp_k, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)
    config[i, t] = -config[i, t]

    bmat_up = dqmc.compute_timestep_mat(exp_k, nu, config, t, +1)
    bmat_dn = dqmc.compute_timestep_mat(exp_k, nu, config, t, -1)
    dqmc.update_timestep_mats(exp_k, nu, config, bmats_up, bmats_dn, t)

    assert_array_equal(bmat_up, bmats_up[t])
    assert_array_equal(bmat_dn, bmats_dn[t])


@given(st_u, st_beta, st_nsites, st_ntimes, st_i, st_t)
def test_update(u, beta, num_sites, num_times, i, t):
    mu = u/2
    assume(i < num_sites)
    assume(t < num_times)
    exp_k, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    old_spin = config[i, t]
    dqmc.update(exp_k, nu, config, bmats_up, bmats_dn, i, t)

    bmats_up_new, bmats_dn_new = dqmc.compute_timestep_mats(exp_k, nu, config)

    assert config[i, t] == -old_spin
    assert_array_equal(bmats_up_new, bmats_up)
    assert_array_equal(bmats_dn_new, bmats_dn)


@given(st_u, st_beta, st_nsites, st_ntimes, st_i, st_t)
def test_compute_acceptance_fast(u, beta, num_sites, num_times, i, t):
    mu = u/2
    assume(i < num_sites)
    assume(t < num_times)
    exp_k, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    gf_up, gf_dn, sgns, logdet = dqmc.init_greens(bmats_up, bmats_dn, t)
    # gf_up, gf_dn = dqmc.compute_greens(bmats_up, bmats_dn, t)
    # Compute fast acceptance *before* spin flip
    d = dqmc.compute_acceptance_fast(nu, config, gf_up, gf_dn, i, t)

    # Compute old determinants before flip
    m_up, m_dn = dqmc.compute_m_matrices(bmats_up, bmats_dn, t)
    det_up = la.det(m_up)
    det_dn = la.det(m_dn)
    old_det = abs(det_up * det_dn)
    # Flip spin
    dqmc.update(exp_k, nu, config, bmats_up, bmats_dn, i, t)
    # Compute acceptance ratio
    m_up, m_dn = dqmc.compute_m_matrices(bmats_up, bmats_dn, t)
    det_up = la.det(m_up)
    det_dn = la.det(m_dn)
    new_det = abs(det_up * det_dn)
    d_slow = min(new_det / old_det, 1.0)
    assert abs(d_slow - d) < 1e-6


def test_compute_greens():
    pass


@given(st_u, st_beta, st_nsites, st_ntimes, st_t, st.integers(1, 20))
def test_compute_greens_stable(u, beta, num_sites, num_times, t, prod_len):
    mu = u/2
    assume(num_times % prod_len == 0)
    assume(t < num_times)
    exp_k, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    # Compute Green's function of time slice `t`
    gf_up_ref, gf_dn_ref, sgn_ref, det_ref = dqmc.init_greens(bmats_up, bmats_dn, t)

    # Compute stable Green's function of time slice `t`
    gf_up, gf_dn, sgn, det = dqmc.init_greens(bmats_up, bmats_dn, t, prod_len)

    assert_allclose(gf_up, gf_up_ref, rtol=1e-6, atol=10)
    assert_allclose(gf_dn, gf_dn_ref, rtol=1e-6, atol=10)
    assert_array_equal(sgn_ref, sgn)
    assert_allclose(det_ref, det)


@given(st_u, st_beta, st_nsites, st_ntimes, st_i)
def test_update_greens(u, beta, num_sites, num_times, i):
    mu = u/2
    assume(i < num_sites)
    exp_k, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    t = 0  # Only use first time slice to prevent need for wrapping
    gf_up, gf_dn, _, _ = dqmc.init_greens(bmats_up, bmats_dn, t)

    # Update Green's function
    dqmc.update_greens(nu, config, gf_up, gf_dn, i, t)
    # Update configuration and B-matrices and compute Greens function
    dqmc.update(exp_k, nu, config, bmats_up, bmats_dn, i, t)
    gf_up_ref, gf_dn_ref, _, _ = dqmc.init_greens(bmats_up, bmats_dn, t)

    assert_array_almost_equal(gf_up, gf_up_ref, decimal=8)
    assert_array_almost_equal(gf_dn, gf_dn_ref, decimal=8)


@given(st_u, st_beta, st_nsites, st_ntimes, st_i)
def test_update_greens_blas(u, beta, num_sites, num_times, i):
    mu = u/2
    assume(i < num_sites)
    exp_k, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    t = 0  # Only use first time slice to prevent need for wrapping
    gf_up, gf_dn, _, _ = dqmc.init_greens(bmats_up, bmats_dn, t)
    # Update Green's function
    dqmc.update_greens_blas(nu, config, gf_up, gf_dn, i, t)
    # Update configuration and B-matrices and compute Greens function
    dqmc.update(exp_k, nu, config, bmats_up, bmats_dn, i, t)
    gf_up_ref, gf_dn_ref, _, _ = dqmc.init_greens(bmats_up, bmats_dn, t)

    assert_array_almost_equal(gf_up, gf_up_ref, decimal=8)
    assert_array_almost_equal(gf_dn, gf_dn_ref, decimal=8)


@given(st_u, st_beta, st_nsites, st_ntimes, st_t)
def test_wrap_up_greens(u, beta, num_sites, num_times, t):
    mu = u/2
    assume(t < num_times)
    exp_k, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    # Compute Green's function of time slice `t`
    try:
        gf_up, gf_dn, _, _ = dqmc.init_greens(bmats_up, bmats_dn, t)
    except la.LinAlgError:
        assume(False)
        return
    # Wrap Greens function to next time slice `t+1`
    dqmc.wrap_up_greens(bmats_up, bmats_dn, gf_up, gf_dn, t)
    # Re-compute Green's function for next time slice `t+1`
    gf_up_ref, gf_dn_ref, _, _ = dqmc.init_greens(bmats_up, bmats_dn, t+1)

    assert_array_almost_equal(gf_up_ref, gf_up, decimal=5)
    assert_array_almost_equal(gf_dn_ref, gf_dn, decimal=5)


@given(st_u, st_beta, st_nsites, st_ntimes, st_t)
def test_wrap_down_greens(u, beta, num_sites, num_times, t):
    mu = u/2
    assume(t < num_times)
    exp_k, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    # Compute Green's function of time slice `t`
    gf_up, gf_dn, _, _ = dqmc.init_greens(bmats_up, bmats_dn, t)

    # Wrap Greens function to next time slice `t+1`
    dqmc.wrap_down_greens(bmats_up, bmats_dn, gf_up, gf_dn, t)

    # Re-compute Green's function for next time slice `t+1`
    gf_up_ref, gf_dn_ref, _, _ = dqmc.init_greens(bmats_up, bmats_dn, t-1)

    assert_array_almost_equal(gf_up_ref, gf_up, decimal=5)
    assert_array_almost_equal(gf_dn_ref, gf_dn, decimal=5)
