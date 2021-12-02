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


def test_init_qmc():
    pass


@given(st.integers(2, 10), st.integers(1, 10))
def test_greens_function_update(u, beta):
    num_sites = 5
    i = random.randint(0, num_sites - 1)
    t = 0

    model = hubbard_hypercube(5, u=u, mu=0, beta=beta / 10, periodic=0)
    num_timesteps = 200

    exp_k, nu, config = dqmc.init_qmc(model, num_timesteps)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(exp_k, nu, config)

    # Initial order: L, L-1, ..., 2, 1
    bmat_order = np.arange(config.shape[1], dtype=np.int64)[::-1]
    gf_up, gf_dn = dqmc.compute_greens(bmats_up, bmats_dn, bmat_order)

    # Update Green's function
    # -----------------------
    dqmc.update_greens2(nu, config, gf_up, gf_dn, i, t)
    # -----------------------

    # Update field
    config[i, t] = -config[i, t]
    dqmc.update_timestep_mats(exp_k, nu, config, bmats_up, bmats_dn, t)

    # Compare to exact result
    gf_up_ref, gf_dn_ref = dqmc.compute_greens(bmats_up, bmats_dn, bmat_order)
    assert_array_almost_equal(gf_up, gf_up_ref, decimal=4)
    assert_array_almost_equal(gf_dn, gf_dn_ref, decimal=4)
