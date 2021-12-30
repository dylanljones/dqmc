# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from . import _timeflow as _tf  # noqa


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
