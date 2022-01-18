# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Tests for the main dqmc methods.

The deadline of the hypothesis tests have been increased to prevent `flaky` errors.
"""

import numpy as np
import scipy.linalg as la
from numpy.testing import assert_equal, assert_allclose
from hypothesis import given, settings, assume, strategies as st
from dqmc import dqmc, hubbard_hypercube

settings.load_profile("dqmc")

st_u = st.floats(0.1, 10)
st_beta = st.floats(0.01, 7)
st_mu = st.floats(0, 5)
st_nsites = st.integers(4, 20)
st_ntimes = st.integers(50, 200).filter(lambda x: x % 2 == 0)
st_i = st.integers(0, 20)
st_t = st.integers(0, 200)
st_nprod = st.integers(2, 20).filter(lambda x: x % 2 == 0)


def _init(num_sites, num_timesteps, u, mu, beta, eps=0.0, periodic=True):
    hop = 1.0
    assume(u * hop * (beta / num_timesteps)**2 < 0.1)
    model = hubbard_hypercube(num_sites, u, eps, hop, mu, beta, periodic=periodic)
    expk, _, nu, config = dqmc.init_qmc(model, num_timesteps, 0)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(expk, nu, config)
    return expk, nu, config, bmats_up, bmats_dn


def _greens(bmats_up, bmats_dn, t, prod_len=0):
    try:
        return dqmc.init_greens(bmats_up, bmats_dn, t, prod_len)
    except la.LinAlgError:
        assume(False)
    gf = np.zeros(bmats_up.shape[0], dtype=np.float64)
    return gf, np.copy(gf), 0., 0.


def assert_gf_equal(actual, desired, rtol=1e-8, atol=1e-5):
    assert_allclose(actual, desired, rtol=rtol, atol=atol)


def test_init_qmc_atomic():
    model = hubbard_hypercube(5, u=1, hop=0., beta=1.0)
    expk, _, nu, config = dqmc.init_qmc(model, 100, 0)
    assert_equal(np.eye(model.num_sites), expk)
    assert abs(nu - 0.1) < 1e-3


def test_init_qmc_noninter():
    model = hubbard_hypercube(5, u=0, hop=1., beta=1.0)
    expk, _, nu, config = dqmc.init_qmc(model, 100, 0)

    expected = np.eye(model.num_sites)
    np.fill_diagonal(expected[1:, :], -0.01)
    np.fill_diagonal(expected[:, 1:], -0.01)
    assert_allclose(expected, expk, atol=1e-3)
    assert nu == 0


def test_init_qmc_zerot():
    model = hubbard_hypercube(5, u=1, hop=1., beta=0.0)
    expk, _, nu, config = dqmc.init_qmc(model, 100, 0)
    assert_equal(np.eye(model.num_sites), expk)
    assert nu == 0


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_t)
def test_compute_timestep_mat(u, mu, beta, num_sites, num_times, t):
    eps, hop = 0.0, 1.0
    assume(t < num_times)
    assume(u * hop * (beta / num_times)**2 < 0.1)
    model = hubbard_hypercube(num_sites, u, eps, hop, mu, beta, periodic=True)
    expk, _, nu, config = dqmc.init_qmc(model, num_times, 0)

    # Spin up
    sigma = +1
    diag = sigma * nu * config[:, t]
    expected = np.dot(expk, la.expm(np.diag(diag)))
    result = dqmc.compute_timestep_mat(expk, nu, config, t, sigma)
    assert_allclose(expected, result, rtol=1e-8)

    # Spin down
    sigma = -1
    diag = sigma * nu * config[:, t]
    expected = np.dot(expk, la.expm(np.diag(diag)))
    result = dqmc.compute_timestep_mat(expk, nu, config, t, sigma)
    assert_allclose(expected, result, rtol=1e-8)


# Keep getting Flaky errors (unuesd anyway)
# @given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_t)
# def test_compute_timestep_mat_inv(u, mu, beta, num_sites, num_times, t):
#     hop = 1.0
#     assume(u * hop * (beta / num_times)**2 < 0.1)
#     model = hubbard_hypercube(num_sites, u, 0.0, hop, mu, beta, periodic=True)
#     expk, expk_inv, nu, config = dqmc.init_qmc(model, num_times, 0)
#
#     bmat_up = dqmc.compute_timestep_mat(expk, nu, config, t, +1)
#     bmat_dn = dqmc.compute_timestep_mat(expk, nu, config, t, -1)
#
#     bmat_up_inv = dqmc.compute_timestep_mat_inv(expk_inv, nu, config, t, +1)
#     bmat_dn_inv = dqmc.compute_timestep_mat_inv(expk_inv, nu, config, t, -1)
#
#     assert_allclose(la.inv(bmat_up), bmat_up_inv, rtol=1e-8)
#     assert_allclose(la.inv(bmat_dn), bmat_dn_inv, rtol=1e-8)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes)
def test_compute_timestep_mats(u, mu, beta, num_sites, num_times):
    expk, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    # Check order of B-matrices
    t1 = num_times - 1
    first_up = dqmc.compute_timestep_mat(expk, nu, config, t=0, sigma=+1)
    first_dn = dqmc.compute_timestep_mat(expk, nu, config, t=0, sigma=-1)
    last_up = dqmc.compute_timestep_mat(expk, nu, config, t=t1, sigma=+1)
    last_dn = dqmc.compute_timestep_mat(expk, nu, config, t=t1, sigma=-1)
    assert_equal(last_up, bmats_up[-1])
    assert_equal(last_dn, bmats_dn[-1])
    assert_equal(first_up, bmats_up[0])
    assert_equal(first_dn, bmats_dn[0])


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_i, st_t)
def test_update_timestep_mats(u, mu, beta, num_sites, num_times, i, t):
    assume((i < num_sites) and (t < num_times))
    expk, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)
    config[i, t] = -config[i, t]

    bmat_up = dqmc.compute_timestep_mat(expk, nu, config, t, +1)
    bmat_dn = dqmc.compute_timestep_mat(expk, nu, config, t, -1)
    dqmc.update_timestep_mats(expk, nu, config, bmats_up, bmats_dn, t)

    assert_equal(bmat_up, bmats_up[t])
    assert_equal(bmat_dn, bmats_dn[t])


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_i, st_t)
def test_update(u, mu, beta, num_sites, num_times, i, t):
    assume((i < num_sites) and (t < num_times))
    expk, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    old_spin = config[i, t]
    dqmc.update(expk, nu, config, bmats_up, bmats_dn, i, t)

    bmats_up_new, bmats_dn_new = dqmc.compute_timestep_mats(expk, nu, config)

    assert config[i, t] == -old_spin
    assert_equal(bmats_up_new, bmats_up)
    assert_equal(bmats_dn_new, bmats_dn)


@given(st.floats(0.1, 6), st_mu, st_beta, st_nsites, st_ntimes, st_i, st_t)
def test_compute_acceptance_fast(u, mu, beta, num_sites, num_times, i, t):
    """Test computation of fast acceptance ratio.

    Instabilities grow with U, so limit maximum U to prevent too big instabilities
    in non-stabilized method.
    """
    prod_len = 0
    assume((i < num_sites) and (t < num_times))
    assume(prod_len == 0 or num_times % prod_len == 0)
    expk, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)
    gf_up, gf_dn, sgns, logdet = _greens(bmats_up, bmats_dn, t, prod_len)

    # Compute fast acceptance *before* spin flip
    d = dqmc.compute_acceptance_fast(nu, config, gf_up, gf_dn, i, t)
    # Compute old determinants before flip
    m_up, m_dn = dqmc.compute_m_matrices(bmats_up, bmats_dn, t)
    det_up, det_dn = 1., 1.
    try:
        det_up, det_dn = la.det(m_up), la.det(m_dn)
    except la.LinAlgError:
        assume(False)
    old_det = abs(det_up * det_dn)
    assume(old_det != 0)

    # Flip spin
    dqmc.update(expk, nu, config, bmats_up, bmats_dn, i, t)
    # Compute acceptance ratio
    m_up, m_dn = dqmc.compute_m_matrices(bmats_up, bmats_dn, t)
    try:
        det_up, det_dn = la.det(m_up), la.det(m_dn)
    except la.LinAlgError:
        assume(False)
    new_det = abs(det_up * det_dn)
    d_slow = min(new_det / old_det, 1.0)
    assert abs(d_slow - d) < 0.05


@given(st.floats(0.1, 6), st_mu, st_beta, st_nsites, st_ntimes, st_t, st_nprod)
def test_compute_greens_stable(u, mu, beta, num_sites, num_times, t, prod_len):
    """Test stabilized computation of Green's function.

    Instabilities grow with U, so limit maximum U to prevent too big instabilities
    in non-stabilized method.
    """
    assume(num_times % prod_len == 0)
    assume(t < num_times)
    expk, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    # Compute Green's function of time slice `t`
    gf_up_ref, gf_dn_ref, sgn_ref, det_ref = _greens(bmats_up, bmats_dn, t)
    # Compute stable Green's function of time slice `t`
    gf_up, gf_dn, sgn, det = _greens(bmats_up, bmats_dn, t, prod_len)
    rtol = 0.5

    assert_gf_equal(gf_up, gf_up_ref, atol=0.05, rtol=rtol)
    assert_gf_equal(gf_dn, gf_dn_ref, atol=0.05, rtol=rtol)
    assert_equal(sgn_ref, sgn)
    assert_allclose(det_ref, det, rtol=0.1)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_i, st_t)
def test_update_greens(u, mu, beta, num_sites, num_times, i, t):
    prod_len = 2
    assume((i < num_sites) and (t < num_times))
    assume(num_times % prod_len == 0)
    expk, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)
    gf_up, gf_dn, _, _ = _greens(bmats_up, bmats_dn, t, prod_len)

    # Update Green's function
    dqmc.update_greens(nu, config, gf_up, gf_dn, i, t)
    # Update configuration and B-matrices and compute Greens function
    dqmc.update(expk, nu, config, bmats_up, bmats_dn, i, t)
    gf_up_ref, gf_dn_ref, _, _ = _greens(bmats_up, bmats_dn, t, prod_len)

    assert_gf_equal(gf_up, gf_up_ref)
    assert_gf_equal(gf_dn, gf_dn_ref)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_i, st_t)
def test_update_greens_blas(u, mu, beta, num_sites, num_times, i, t):
    prod_len = 2
    assume((i < num_sites) and (t < num_times))
    assume(num_times % prod_len == 0)
    expk, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)
    gf_up, gf_dn, _, _ = _greens(bmats_up, bmats_dn, t, prod_len)

    # Update Green's function
    dqmc.update_greens_blas(nu, config, gf_up, gf_dn, i, t)
    # Update configuration and B-matrices and compute Greens function
    dqmc.update(expk, nu, config, bmats_up, bmats_dn, i, t)
    gf_up_ref, gf_dn_ref, _, _ = _greens(bmats_up, bmats_dn, t, prod_len)

    assert_gf_equal(gf_up, gf_up_ref)
    assert_gf_equal(gf_dn, gf_dn_ref)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_t)
def test_wrap_up_greens(u, mu, beta, num_sites, num_times, t):
    prod_len = 2
    assume(t < num_times)
    assume(num_times % prod_len == 0)
    expk, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    # Compute Green's function of time slice `t`
    gf_up, gf_dn, _, _ = _greens(bmats_up, bmats_dn, t, prod_len)
    # Wrap Greens function to next time slice `t+1`
    dqmc.wrap_up_greens(bmats_up, bmats_dn, gf_up, gf_dn, t)
    # Re-compute Green's function for next time slice `t+1`
    tp1 = (t + 1) % num_times
    gf_up_ref, gf_dn_ref, _, _ = _greens(bmats_up, bmats_dn, tp1, prod_len)

    assert_gf_equal(gf_up, gf_up_ref)
    assert_gf_equal(gf_dn, gf_dn_ref)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_t)
def test_wrap_down_greens(u, mu, beta, num_sites, num_times, t):
    prod_len = 2
    assume(t < num_times)
    assume(num_times % prod_len == 0)
    expk, nu, config, bmats_up, bmats_dn = _init(num_sites, num_times, u, mu, beta)

    # Compute Green's function of time slice `t`
    gf_up, gf_dn, _, _ = _greens(bmats_up, bmats_dn, t, prod_len)
    # Wrap Greens function to next time slice `t+1`
    dqmc.wrap_down_greens(bmats_up, bmats_dn, gf_up, gf_dn, t)
    # Re-compute Green's function for next time slice `t+1`
    tm1 = t - 1 if t > 0 else num_times - 1
    gf_up_ref, gf_dn_ref, _, _ = _greens(bmats_up, bmats_dn, tm1, prod_len)

    assert_gf_equal(gf_up, gf_up_ref)
    assert_gf_equal(gf_dn, gf_dn_ref)
