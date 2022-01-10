# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from numpy.testing import assert_allclose
from hypothesis import given, settings, assume, strategies as st
from dqmc import Parameters, run_dqmc

settings.register_profile("sim", deadline=None, max_examples=5,
                          report_multiple_bugs=True)
settings.load_profile("sim")

st_u = st.floats(0.1, 10)
st_beta = st.floats(0.01, 7)
st_mu = st.floats(0, 5)
st_nsites = st.integers(4, 20)
st_ntimes = st.integers(50, 200).filter(lambda x: x % 8 == 0)


def _init(u, mu, beta, num_sites=6, num_times=40, nequil=512, nsampl=512,
          nwraps=8, prod_len=8, recomp=1):
    dt = beta / num_times
    eps, hop = 0.0, 1.0
    assume(u * hop * (beta / num_times) ** 2 < 0.1)
    p = Parameters(num_sites, u, eps, hop, mu, dt, num_times,
                   nequil, nsampl, nwraps, recomp, prod_len)
    return p


@given(st_u)
def test_half_filling_t1(u):
    mu = 0.0
    temp = 1.0
    p = _init(u, mu, 1 / temp)
    n_up, n_dn, n_dbl, m2, _ = run_dqmc(p)
    assert_allclose(n_up + n_dn, np.full_like(n_up, fill_value=1.0), rtol=0.01)


@given(st_u)
def test_half_filling_tlow(u):
    mu = 0.0
    temp = 0.1
    p = _init(u, mu, 1 / temp)
    n_up, n_dn, n_dbl, m2, _ = run_dqmc(p)
    assert_allclose(n_up + n_dn, np.full_like(n_up, fill_value=1.0), rtol=0.01)


@given(st_u)
def test_half_filling_thigh(u):
    mu = 0.0
    temp = 5.0
    p = _init(u, mu, 1 / temp)
    n_up, n_dn, n_dbl, m2, _ = run_dqmc(p)
    assert_allclose(n_up + n_dn, np.full_like(n_up, fill_value=1.0), rtol=0.01)
