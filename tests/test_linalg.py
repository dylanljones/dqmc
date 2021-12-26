# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from scipy import linalg as la
from numpy.testing import assert_allclose
from hypothesis import given, assume, strategies as st
import hypothesis.extra.numpy as hnp
from dqmc import linalg
from functools import reduce


xarr = hnp.arrays(dtype=np.float64, shape=10, elements=st.floats(-10., 10))
yarr = hnp.arrays(dtype=np.float64, shape=10, elements=st.floats(-10., 10))
aarr = hnp.arrays(dtype=np.float64, shape=(10, 10), elements=st.floats(-10., 10))


@given(st.floats(-1.0, +1.0), xarr, yarr, aarr)
def test_blas_dger(alpha, x, y, a):
    assume(np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
    assume(np.all(np.isfinite(a)))
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    a = a.astype(np.float64)

    expected = la.blas.blas_dger(alpha, np.copy(x), np.copy(y), a=np.copy(a))
    result = np.copy(a)
    linalg.blas_dger(alpha, x, y, result)
    assert_allclose(expected, result, rtol=1e-10)


@given(st.floats(-1.0, +1.0), xarr, yarr, aarr)
def test_dger_numpy(alpha, x, y, a):
    assume(np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
    assume(np.all(np.isfinite(a)))
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    a = a.astype(np.float64)

    expected = la.blas.blas_dger(alpha, np.copy(x), np.copy(y), a=np.copy(a))
    result = np.copy(a)
    linalg.numpy_dger(alpha, x, y, result)
    assert_allclose(expected, result, rtol=1e-10)
