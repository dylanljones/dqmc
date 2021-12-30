# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from . import _greens  # noqa


def construct_greens(q, d, t, tau, lwork):
    n = q.shape[0]

    q = np.asfortranarray(q)
    d = np.asfortranarray(d)
    t = np.asfortranarray(t)
    tau = np.asfortranarray(tau)
    gf = np.asfortranarray(np.zeros(q.shape, dtype=np.float64))
    sgndet = np.asfortranarray(np.array(0, dtype=np.int32))
    logdet = np.asfortranarray(np.array(0, dtype=np.float64))
    lwork = np.asfortranarray(np.array(lwork, dtype=np.int32))
    info = np.asfortranarray(np.array(0, dtype=np.int32))

    _greens.construct_greens(n, q, d, t, gf, tau, sgndet, logdet, lwork, info)
    return np.ascontiguousarray(gf), int(sgndet), float(logdet)
