# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from . import greens as _gf  # noqa
from . import timeflow as _tf  # noqa

__all__ = [
    "matrix_product_sequence_0beta",
    "matrix_product_sequence_beta0",
    "timeflow_map",
    "timeflow_map_0beta",
    "timeflow_map_beta0",
    "construct_greens"
]


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
    return q, d, t, tau, int(lwork), int(info)


def timeflow_map_0beta(tsm, prod_len, shift):
    timesteps = tsm.shape[0]
    n = tsm.shape[1]
    assert (timesteps % prod_len) == 0
    s = int(timesteps / prod_len)
    shift = shift % timesteps

    tsm = np.asfortranarray(tsm)
    s = np.asfortranarray(np.array(s, dtype=np.int32))
    shift = np.asfortranarray(np.array(shift + 1, dtype=np.int32))
    q = np.asfortranarray(np.zeros((n, n), dtype=np.float64))
    d = np.asfortranarray(np.zeros(n), dtype=np.float64)
    t = np.asfortranarray(np.copy(q), dtype=np.float64)
    tau = np.asfortranarray(np.zeros(n), dtype=np.float64)
    lwork = np.asfortranarray(np.array(0, dtype=np.int32))
    info = np.asfortranarray(np.array(0, dtype=np.int32))

    _tf.timeflow_map_0beta(timesteps, n, tsm, s, shift, q, d, t, tau, lwork, info)
    return q, d, t, tau, int(lwork)


def timeflow_map_beta0(tsm, prod_len, shift):
    timesteps = tsm.shape[0]
    n = tsm.shape[1]
    assert (timesteps % prod_len) == 0
    s = int(timesteps / prod_len)
    shift = shift % timesteps

    tsm = np.asfortranarray(tsm)
    s = np.asfortranarray(np.array(s, dtype=np.int32))
    shift = np.asfortranarray(np.array(shift + 1, dtype=np.int32))
    q = np.asfortranarray(np.zeros((n, n), dtype=np.float64))
    d = np.asfortranarray(np.zeros(n), dtype=np.float64)
    t = np.asfortranarray(np.copy(q), dtype=np.float64)
    tau = np.asfortranarray(np.zeros(n), dtype=np.float64)
    lwork = np.asfortranarray(np.array(0, dtype=np.int32))
    info = np.asfortranarray(np.array(0, dtype=np.int32))

    _tf.timeflow_map_beta0(timesteps, n, tsm, s, shift, q, d, t, tau, lwork, info)
    return q, d, t, tau, int(lwork)


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
