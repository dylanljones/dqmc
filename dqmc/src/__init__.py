# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Python wrappers for the Fortran subroutines."""

import numpy as np
from . import greens as _gf  # noqa
from . import timeflow as _tf  # noqa


# =========================================================================
# Time step matrices / Time flow
# =========================================================================


def compute_timestep_mat(expk, nu, config, t, sigma):
    n, ntimes = config.shape
    config = np.asfortranarray(config, dtype=np.int32)
    b = np.asfortranarray(np.zeros((n, n), dtype=np.float64))
    _tf.compute_timestep_mat(n, ntimes, expk, nu, config, t + 1, b, sigma)
    return b


def compute_timestep_mat_inv(expk_inv, nu, config, t, sigma):
    n, ntimes = config.shape
    config = np.asfortranarray(config, dtype=np.int32)
    b = np.asfortranarray(np.zeros((n, n), dtype=np.float64))
    _tf.compute_timestep_mat_inv(n, ntimes, expk_inv, nu, config, t + 1, b, sigma)
    return b


def compute_timestep_mats(expk, nu, config):
    n, ntimes = config.shape
    tsm_up = np.asfortranarray(np.zeros((ntimes, n, n), dtype=np.float64))
    tsm_dn = np.asfortranarray(np.zeros((ntimes, n, n), dtype=np.float64))
    _tf.compute_timestep_mats(n, ntimes, expk, nu, config, tsm_up, tsm_dn)
    return tsm_up, tsm_dn


def compute_timestep_mats_inv(expk_inv, nu, config):
    n, ntimes = config.shape
    tsm_up = np.asfortranarray(np.zeros((ntimes, n, n), dtype=np.float64))
    tsm_dn = np.asfortranarray(np.zeros((ntimes, n, n), dtype=np.float64))
    _tf.compute_timestep_mats_inv(n, ntimes, expk_inv, nu, config, tsm_up, tsm_dn)
    return tsm_up, tsm_dn


def update_timestep_mats(expk, nu, config, tsm_up, tsm_dn, t):
    n, ntimes = config.shape
    _tf.update_timestep_mats(n, ntimes, expk, nu, config, tsm_up, tsm_dn, t + 1)


def update_timestep_mats_inv(expk_inv, nu, config, tsm_up, tsm_dn, t):
    n, ntimes = config.shape
    _tf.update_timestep_mats_inv(n, ntimes, expk_inv, nu, config,
                                 tsm_up, tsm_dn, t + 1)


def matrix_product_sequence_0beta(mats, prod_len, shift):
    m = mats.shape[0]
    n = mats.shape[1]
    assert (m % prod_len) == 0
    s = int(m / prod_len)
    shift = shift % m
    mats = np.asfortranarray(mats)
    seqs = np.asfortranarray(np.zeros((s, n, n), dtype=np.float64))
    s = np.asfortranarray(np.array(s, dtype=np.int32))
    shift = np.asfortranarray(np.array(shift + 1, dtype=np.int32))
    _tf.matrix_product_sequence_0beta(m, n, mats, s, seqs, shift)
    return seqs


def matrix_product_sequence_beta0(mats, prod_len, shift):
    m = mats.shape[0]
    n = mats.shape[1]
    assert (m % prod_len) == 0
    s = int(m / prod_len)
    shift = shift % m
    mats = np.asfortranarray(mats)
    seqs = np.asfortranarray(np.zeros((s, n, n), dtype=np.float64))
    s = np.asfortranarray(np.array(s, dtype=np.int32))
    shift = np.asfortranarray(np.array(shift + 1, dtype=np.int32))
    _tf.matrix_product_sequence_beta0(m, n, mats, s, seqs, shift)
    return seqs


def timeflow_map(tsm):
    timesteps = tsm.shape[0]
    n = tsm.shape[1]
    tsm = np.asfortranarray(tsm)
    q = np.asfortranarray(np.zeros((n, n), dtype=np.float64))
    d = np.asfortranarray(np.zeros(n), dtype=np.float64)
    t = np.asfortranarray(np.copy(q), dtype=np.float64)
    tau = np.asfortranarray(np.zeros(n), dtype=np.float64)
    lwork = np.asfortranarray(np.array(0, dtype=np.int32))
    info = np.asfortranarray(np.array(0, dtype=np.int32))

    _tf.timeflow_map(timesteps, n, tsm, q, d, t, tau, lwork, info)
    return q, d, t, tau, int(lwork)


def timeflow_map_0beta(tsm, prod_len, shift):
    ntimes = tsm.shape[0]
    n = tsm.shape[1]
    assert (ntimes % prod_len) == 0
    s = int(ntimes / prod_len)
    shift = shift % ntimes

    tsm = np.asfortranarray(tsm)
    s = np.asfortranarray(np.array(s, dtype=np.int32))
    shift = np.asfortranarray(np.array(shift + 1, dtype=np.int32))
    q = np.asfortranarray(np.zeros((n, n), dtype=np.float64))
    d = np.asfortranarray(np.zeros(n), dtype=np.float64)
    t = np.asfortranarray(np.copy(q), dtype=np.float64)
    tau = np.asfortranarray(np.zeros(n), dtype=np.float64)
    lwork = np.asfortranarray(np.array(0, dtype=np.int32))
    info = np.asfortranarray(np.array(0, dtype=np.int32))

    _tf.timeflow_map_0beta(ntimes, n, tsm, s, shift, q, d, t, tau, lwork, info)
    return q, d, t, tau, int(lwork)


def timeflow_map_beta0(tsm, prod_len, shift):
    ntimes = tsm.shape[0]
    n = tsm.shape[1]
    assert (ntimes % prod_len) == 0
    s = int(ntimes / prod_len)
    shift = shift % ntimes

    tsm = np.asfortranarray(tsm)
    s = np.asfortranarray(np.array(s, dtype=np.int32))
    shift = np.asfortranarray(np.array(shift + 1, dtype=np.int32))
    q = np.asfortranarray(np.zeros((n, n), dtype=np.float64))
    d = np.asfortranarray(np.zeros(n), dtype=np.float64)
    t = np.asfortranarray(np.copy(q), dtype=np.float64)
    tau = np.asfortranarray(np.zeros(n), dtype=np.float64)
    lwork = np.asfortranarray(np.array(0, dtype=np.int32))
    info = np.asfortranarray(np.array(0, dtype=np.int32))

    _tf.timeflow_map_beta0(ntimes, n, tsm, s, shift, q, d, t, tau, lwork, info)
    return q, d, t, tau, int(lwork)


# =========================================================================
# Greens
# =========================================================================


def construct_greens(tsm, gf):
    timesteps = tsm.shape[0]
    n = tsm.shape[1]
    tsm = np.asfortranarray(tsm)
    q = np.asfortranarray(np.zeros((n, n), dtype=np.float64))
    d = np.asfortranarray(np.zeros(n), dtype=np.float64)
    t = np.asfortranarray(np.copy(q), dtype=np.float64)
    tau = np.asfortranarray(np.zeros(n), dtype=np.float64)
    lwork = np.asfortranarray(np.array(0, dtype=np.int32))
    info = np.asfortranarray(np.array(0, dtype=np.int32))
    gf_out = np.asfortranarray(np.zeros(q.shape, dtype=np.float64))
    sgndet = np.asfortranarray(np.array(0, dtype=np.int32))
    logdet = np.asfortranarray(np.array(0, dtype=np.float64))

    _tf.timeflow_map(timesteps, n, tsm, q, d, t, tau, lwork, info)

    _gf.construct_greens(n, q, d, t, gf_out, tau, sgndet, logdet, lwork, info)
    gf[:, :] = gf_out
    return int(sgndet), float(logdet)


def update_greens(nu, config, gf_up, gf_dn, i, t):
    n, ntimes = config.shape
    config = np.asfortranarray(config, dtype=np.int32)
    _gf.update_greens(ntimes, n, nu, config, gf_up, gf_dn, i + 1, t + 1)
    return gf_up, gf_dn


def wrap_up_greens(tsm, tsm_inv, gf, t):
    ntimes, n, _ = tsm.shape
    _gf.wrap_up_greens(ntimes, n, tsm, tsm_inv, gf, t + 1)


def wrap_down_greens(tsm, tsm_inv, gf, t):
    ntimes, n, _ = tsm.shape
    _gf.wrap_down_greens(ntimes, n, tsm, tsm_inv, gf, t + 1)
