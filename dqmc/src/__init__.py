# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

from .qrp import qrp as _qrp  # noqa
import numpy as np
from numba import njit


def qrp(a):
    # perform Qr decomposition with column pivoting using Fortran implementation
    m, n = a.shape
    jpvt = np.asfortranarray(np.zeros(n, dtype=np.int32))
    q = np.asfortranarray(np.copy(a))
    r = np.asfortranarray(np.zeros_like(a))
    _qrp(m, n, q, r, jpvt)
    # Construct permutation matrix
    p = np.zeros(q.shape, dtype=np.int64)
    p[jpvt-1, np.arange(n)] = 1
    return q, r, p
