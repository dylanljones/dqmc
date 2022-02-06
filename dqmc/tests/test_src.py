# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import pytest
import numpy as np
import scipy.linalg as la
from numpy.testing import assert_equal, assert_allclose
from hypothesis import given, settings, assume, strategies as st
import hypothesis.extra.numpy as hnp
from dqmc import dqmc, linalg, hubbard_hypercube

# Try to import Fortran implementation
try:
    from dqmc.src import construct_greens
    from dqmc import src
    _fortran_available = True
except ImportError:
    _fortran_available = False
    src = None

settings.load_profile("dqmc")

st_u = st.floats(0.1, 10)
st_beta = st.floats(0.01, 7)
st_mu = st.floats(0, 5)
st_nsites = st.integers(4, 20)
st_ntimes = st.integers(50, 200).filter(lambda x: x % 2 == 0)
st_i = st.integers(0, 20)
st_t = st.integers(0, 200)
st_nprod = st.integers(2, 20).filter(lambda x: x % 2 == 0)
st_tsm = hnp.arrays(np.float64, st.tuples(
                        st.integers(10, 40),
                        st.shared(st.integers(2, 10), key="n"),
                        st.shared(st.integers(2, 10), key="n"),
                    ),
                    elements=st.floats(-1, +1))


def _init(num_sites, num_timesteps, u, mu, beta, eps=0.0, periodic=True):
    hop = 1.0
    assume(u * hop * (beta / num_timesteps)**2 < 0.1)
    model = hubbard_hypercube(num_sites, u, eps, hop, mu, beta, periodic=periodic)
    expk, expk_inv, nu, config = dqmc.init_qmc(model, num_timesteps, 0)
    tsm_up, tsm_dn = dqmc.compute_timestep_mats(expk, nu, config)
    return expk, expk_inv, nu, config, tsm_up, tsm_dn


def _greens(tsm_up, tsm_dn, t, prod_len=0):
    try:
        return dqmc.init_greens(tsm_up, tsm_dn, t, prod_len)
    except la.LinAlgError:
        assume(False)
    gf = np.zeros(tsm_up.shape[0], dtype=np.float64)
    return gf, np.copy(gf), 0., 0.


def assert_gf_equal(actual, desired, rtol=1e-8, atol=1e-5):
    assert_allclose(actual, desired, rtol=rtol, atol=atol)


def assume_fortran():
    if not _fortran_available:
        pytest.skip()


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_t)
def test_compute_timestep_mat(u, mu, beta, num_sites, num_times, t):
    assume_fortran()
    eps, hop = 0.0, 1.0
    assume(t < num_times)
    assume(u * hop * (beta / num_times) ** 2 < 0.1)
    model = hubbard_hypercube(num_sites, u, eps, hop, mu, beta, periodic=True)
    expk, _, nu, config = dqmc.init_qmc(model, num_times, 0)

    b_up_ref = dqmc.compute_timestep_mat(expk, nu, config, t, +1)
    b_dn_ref = dqmc.compute_timestep_mat(expk, nu, config, t, -1)

    b_up = src.compute_timestep_mat(expk, nu, config, t, +1)
    b_dn = src.compute_timestep_mat(expk, nu, config, t, -1)

    assert_allclose(b_up, b_up_ref, rtol=1e-8)
    assert_allclose(b_dn, b_dn_ref, rtol=1e-8)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_t)
def test_compute_timestep_mat_inv(u, mu, beta, num_sites, num_times, t):
    assume_fortran()
    eps, hop = 0.0, 1.0
    assume(t < num_times)
    assume(u * hop * (beta / num_times) ** 2 < 0.1)
    model = hubbard_hypercube(num_sites, u, eps, hop, mu, beta, periodic=True)
    _, expk_inv, nu, config = dqmc.init_qmc(model, num_times, 0)

    b_up_ref = dqmc.compute_timestep_mat_inv(expk_inv, nu, config, t, +1)
    b_dn_ref = dqmc.compute_timestep_mat_inv(expk_inv, nu, config, t, -1)

    b_up = src.compute_timestep_mat_inv(expk_inv, nu, config, t, +1)
    b_dn = src.compute_timestep_mat_inv(expk_inv, nu, config, t, -1)

    assert_allclose(b_up, b_up_ref, rtol=1e-8)
    assert_allclose(b_dn, b_dn_ref, rtol=1e-8)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes)
def test_compute_timestep_mats(u, mu, beta, nsites, ntimes):
    assume_fortran()
    eps, hop = 0.0, 1.0
    assume(u * hop * (beta / ntimes) ** 2 < 0.1)
    model = hubbard_hypercube(nsites, u, eps, hop, mu, beta, periodic=True)
    expk, _, nu, config = dqmc.init_qmc(model, ntimes, 0)

    tsm_up_ref, tsm_dn_ref = dqmc.compute_timestep_mats(expk, nu, config)
    tsm_up, tsm_dn = src.compute_timestep_mats(expk, nu, config)

    assert_allclose(tsm_up, tsm_up_ref, rtol=1e-8)
    assert_allclose(tsm_dn, tsm_dn_ref, rtol=1e-8)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes)
def test_compute_timestep_mats_inv(u, mu, beta, nsites, ntimes):
    assume_fortran()
    eps, hop = 0.0, 1.0
    assume(u * hop * (beta / ntimes) ** 2 < 0.1)
    model = hubbard_hypercube(nsites, u, eps, hop, mu, beta, periodic=True)
    _, expk_inv, nu, config = dqmc.init_qmc(model, ntimes, 0)

    tsm_up_ref, tsm_dn_ref = dqmc.compute_timestep_mats_inv(expk_inv, nu, config)
    tsm_up, tsm_dn = src.compute_timestep_mats_inv(expk_inv, nu, config)

    assert_allclose(tsm_up, tsm_up_ref, rtol=1e-8)
    assert_allclose(tsm_dn, tsm_dn_ref, rtol=1e-8)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_i, st_t)
def test_update_timestep_mats(u, mu, beta, nsites, ntimes, i, t):
    assume_fortran()
    assume((i < nsites) and (t < ntimes))
    expk, _, nu, config, tsm_up, tsm_dn = _init(nsites, ntimes, u, mu, beta)
    tsm_up, tsm_dn = np.asfortranarray(tsm_up), np.asfortranarray(tsm_dn)
    tsm_up_ref = np.ascontiguousarray(tsm_up)
    tsm_dn_ref = np.ascontiguousarray(tsm_dn)

    # Update config
    config[i, t] = -config[i, t]
    # Update time step mats
    dqmc.update_timestep_mats(expk, nu, config, tsm_up_ref, tsm_dn_ref, t)
    src.update_timestep_mats(expk, nu, config, tsm_up, tsm_dn, t)

    assert_allclose(tsm_up, tsm_up_ref, rtol=1e-8)
    assert_allclose(tsm_dn, tsm_dn_ref, rtol=1e-8)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_i, st_t)
def test_update_timestep_mats_inv(u, mu, beta, nsites, ntimes, i, t):
    assume_fortran()
    assume((i < nsites) and (t < ntimes))
    _, expk_inv, nu, config, tsm_up, tsm_dn = _init(nsites, ntimes, u, mu, beta)
    tsm_up, tsm_dn = np.asfortranarray(tsm_up), np.asfortranarray(tsm_dn)
    tsm_up_ref = np.ascontiguousarray(tsm_up)
    tsm_dn_ref = np.ascontiguousarray(tsm_dn)

    # Update config
    config[i, t] = -config[i, t]
    # Update time step mats
    dqmc.update_timestep_mats_inv(expk_inv, nu, config, tsm_up_ref, tsm_dn_ref, t)
    src.update_timestep_mats_inv(expk_inv, nu, config, tsm_up, tsm_dn, t)

    assert_allclose(tsm_up, tsm_up_ref, rtol=1e-8)
    assert_allclose(tsm_dn, tsm_dn_ref, rtol=1e-8)


@given(st_tsm, st.integers(1, 16), st.integers(0, 10))
def test_matrix_product_sequence_0beta(mats, prod_len, shift):
    assume_fortran()
    num_mats = len(mats)
    assume(num_mats % prod_len == 0)
    assume(np.all(np.isfinite(mats)))
    assume(shift < num_mats)

    seq_ref = linalg.matrix_product_sequence_0beta(mats, prod_len, shift)
    seq = src.matrix_product_sequence_0beta(mats, prod_len, shift)

    assert_allclose(seq, seq_ref, atol=1e-10)


@given(st_tsm, st.integers(1, 16), st.integers(0, 10))
def test_matrix_product_sequence_beta0(mats, prod_len, shift):
    assume_fortran()
    num_mats = len(mats)
    assume(num_mats % prod_len == 0)
    assume(np.all(np.isfinite(mats)))
    assume(shift < num_mats)

    seq_ref = linalg.matrix_product_sequence_beta0(mats, prod_len, shift)
    seq = src.matrix_product_sequence_beta0(mats, prod_len, shift)

    assert_allclose(seq, seq_ref, atol=1e-10)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_t, st_nprod)
def test_compute_greens(u, mu, beta, num_sites, num_times, t, prod_len):
    assume_fortran()
    assume(num_times % prod_len == 0)
    assume(t < num_times)
    expk, _, nu, config, tsm_up, tsm_dn = _init(num_sites, num_times, u, mu, beta)

    # Compute Green's function of time slice `t`
    gf_up_ref, gf_dn_ref, sgn_ref, det_ref = _greens(tsm_up, tsm_dn, t)

    shape = (num_sites, num_sites)
    gf_up = np.asfortranarray(np.zeros(shape, dtype=np.float64))
    gf_dn = np.asfortranarray(np.zeros(shape, dtype=np.float64))

    tsm_up = dqmc.matrix_product_sequence_0beta(tsm_up, prod_len, t)
    tsm_dn = dqmc.matrix_product_sequence_0beta(tsm_dn, prod_len, t)

    sgn, det = np.zeros(2), np.zeros(2)
    sgn[0], det[0] = src.construct_greens(tsm_up, gf_up)
    sgn[1], det[1] = src.construct_greens(tsm_dn, gf_dn)

    rtol = 0.5

    assert_gf_equal(gf_up, gf_up_ref, atol=0.05, rtol=rtol)
    assert_gf_equal(gf_dn, gf_dn_ref, atol=0.05, rtol=rtol)
    assert_equal(sgn_ref, sgn)
    assert_allclose(det_ref, det, rtol=0.1)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_i, st_t)
def test_update_greens(u, mu, beta, num_sites, num_times, i, t):
    assume_fortran()
    prod_len = 2
    assume((i < num_sites) and (t < num_times))
    assume(num_times % prod_len == 0)

    expk, _, nu, config, tsm_up, tsm_dn = _init(num_sites, num_times, u, mu, beta)
    gf_up_ref, gf_dn_ref, _, _ = _greens(tsm_up, tsm_dn, t, prod_len)
    gf_up = np.asfortranarray(gf_up_ref)
    gf_dn = np.asfortranarray(gf_dn_ref)

    dqmc.update_greens(nu, config, gf_up_ref, gf_dn_ref, i, t)
    src.update_greens(nu, config, gf_up, gf_dn, i, t)

    assert_gf_equal(gf_up, gf_up_ref)
    assert_gf_equal(gf_dn, gf_dn_ref)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_t)
def test_wrap_up_greens(u, mu, beta, nsites, ntimes, t):
    assume_fortran()
    prod_len = 2
    assume((t < ntimes) and (ntimes % prod_len == 0))
    expk, expk_inv, nu, config, tsm_up, tsm_dn = _init(nsites, ntimes, u, mu, beta)
    tsm_up_inv, tsm_dn_inv = dqmc.compute_timestep_mats_inv(expk_inv, nu, config)
    gf_up_ref, gf_dn_ref, _, _ = _greens(tsm_up, tsm_dn, t, prod_len)
    gf_up = np.asfortranarray(gf_up_ref)
    gf_dn = np.asfortranarray(gf_dn_ref)

    dqmc.wrap_up_greens(tsm_up, tsm_dn, gf_up_ref, gf_dn_ref, t)
    src.wrap_up_greens(tsm_up, tsm_up_inv, gf_up, t)
    src.wrap_up_greens(tsm_dn, tsm_dn_inv, gf_dn, t)

    assert_gf_equal(gf_up, gf_up_ref)
    assert_gf_equal(gf_dn, gf_dn_ref)


@given(st_u, st_mu, st_beta, st_nsites, st_ntimes, st_t)
def test_wrap_down_greens(u, mu, beta, nsites, ntimes, t):
    assume_fortran()
    prod_len = 2
    assume((0 < t < ntimes) and (ntimes % prod_len == 0))
    expk, expk_inv, nu, config, tsm_up, tsm_dn = _init(nsites, ntimes, u, mu, beta)
    tsm_up_inv, tsm_dn_inv = dqmc.compute_timestep_mats_inv(expk_inv, nu, config)
    gf_up_ref, gf_dn_ref, _, _ = _greens(tsm_up, tsm_dn, t, prod_len)
    gf_up = np.asfortranarray(gf_up_ref)
    gf_dn = np.asfortranarray(gf_dn_ref)

    dqmc.wrap_down_greens(tsm_up, tsm_dn, gf_up_ref, gf_dn_ref, t)
    src.wrap_down_greens(tsm_up, tsm_up_inv, gf_up, t)
    src.wrap_down_greens(tsm_dn, tsm_dn_inv, gf_dn, t)

    assert_gf_equal(gf_up, gf_up_ref)
    assert_gf_equal(gf_dn, gf_dn_ref)
