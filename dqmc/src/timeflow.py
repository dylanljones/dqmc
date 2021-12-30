# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from . import _timeflow as _tf  # noqa


def qrp(a):
    n = a.shape[0]

    a = np.asfortranarray(a)
    jpvt = np.asfortranarray(np.zeros(n, dtype=np.int32))
    tau = np.asfortranarray(np.zeros(n), dtype=np.float64)
    lwork = np.asfortranarray(np.array(0, dtype=np.int32))
    info = np.asfortranarray(np.array(0, dtype=np.int32))

    _tf.qrp(n, a, jpvt, tau, lwork, info)

    return a, jpvt, tau, int(lwork), int(info)


def extract_diag(a):
    n = a.shape[0]
    a = np.asfortranarray(a)
    d = np.asfortranarray(np.zeros(n), dtype=np.float64)
    _tf.extract_diag(n, a, d)
    return d


def build_t1(q, d, jpvt):
    n = q.shape[0]
    q = np.asfortranarray(q)
    d = np.asfortranarray(d)
    t = np.asfortranarray(np.zeros((n, n)), dtype=np.float64)
    _tf.build_t1(n, q, jpvt, d, t)
    return t


def build_w(q, d, w, tau, lwork):
    n = q.shape[0]
    q = np.asfortranarray(q)
    d = np.asfortranarray(d)
    w = np.asfortranarray(w)
    tau = np.asfortranarray(tau)
    lwork = np.asfortranarray(np.array(lwork, dtype=np.int32))
    info = np.asfortranarray(np.array(0, dtype=np.int32))
    _tf.build_w(n, q, d, w, tau, lwork, info)
    return w, int(info)


def pre_pivot(w):
    n = w.shape[0]
    w = np.asfortranarray(w)
    jpvt = np.asfortranarray(np.zeros(n, dtype=np.int32))
    _tf.pre_pivot(n, w, jpvt)
    return jpvt


def pre_pivot_qrf(w, lwork):
    n = w.shape[0]
    w = np.asfortranarray(w)
    q = np.asfortranarray(np.zeros_like(w))
    jpvt = np.asfortranarray(np.zeros(n, dtype=np.int32))
    tau = np.asfortranarray(np.zeros(n), dtype=np.float64)
    lwork = np.asfortranarray(np.array(lwork, dtype=np.int32))
    info = np.asfortranarray(np.array(0, dtype=np.int32))
    _tf.pre_pivot_qrf(n, w, q, jpvt, tau, lwork, info)
    return q, jpvt, tau, int(info)


def build_t(q, d, jpvt, t):
    n = q.shape[0]
    q = np.asfortranarray(q)
    d = np.asfortranarray(d)
    jpvt = np.asfortranarray(jpvt)
    t = np.asfortranarray(t)
    _tf.build_t(n, q, jpvt, d, t)
    return t


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
