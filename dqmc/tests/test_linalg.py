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
from scipy import linalg as la
from numpy.testing import assert_allclose
from hypothesis import given, assume, settings, strategies as st
import hypothesis.extra.numpy as hnp
from dqmc import linalg
from functools import reduce

settings.load_profile("dqmc")


xarr = hnp.arrays(np.float64, 10, elements=st.floats(-10., 10))
yarr = hnp.arrays(np.float64, 10, elements=st.floats(-10., 10))
aarr = hnp.arrays(np.float64, (10, 10), elements=st.floats(-10., 10))

mat = hnp.arrays(np.float64,
                 st.tuples(
                    st.shared(st.integers(2, 10), key="n"),
                    st.shared(st.integers(2, 10), key="n"),
                 ),
                 elements=st.floats(-1e10, +1e10))

mat_arr = hnp.arrays(np.float64,
                     st.tuples(
                        st.integers(10, 40),
                        st.shared(st.integers(2, 10), key="n"),
                        st.shared(st.integers(2, 10), key="n"),
                     ),
                     elements=st.floats(-1e5, +1e5))


@given(st.floats(-1.0, +1.0), xarr, yarr, aarr)
def test_blas_dger(alpha, x, y, a):
    assume(np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
    assume(np.all(np.isfinite(a)))
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    a = a.astype(np.float64)

    expected = la.blas.dger(alpha, np.copy(x), np.copy(y), a=np.copy(a))
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

    expected = la.blas.dger(alpha, np.copy(x), np.copy(y), a=np.copy(a))
    result = np.copy(a)
    linalg.numpy_dger(alpha, x, y, result)
    assert_allclose(expected, result, rtol=1e-10)


@given(mat_arr)
def test_mdot(matrices):
    expected = reduce(np.dot, matrices)
    result = linalg.mdot(matrices)
    assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


@given(mat)
def test_decompose_qrp(a):
    try:
        q, r, jpvt = linalg.decompose_qrp(a)
    except Exception:  # noqa
        assume(False)
    else:
        assert_allclose(linalg.reconstruct_qrp(q, r, jpvt), a, rtol=1e-6, atol=10)


@given(mat)
def test_decompose_udt(a):
    try:
        u, d, t = linalg.decompose_udt(a)
    except Exception:  # noqa
        assume(False)
    else:
        rec = linalg.reconstruct_udt(u, d, t)
        assume(np.any(np.isfinite(rec)))
        assert_allclose(rec, a, rtol=1e-6, atol=10)


@given(mat_arr, st.integers(1, 16), st.integers(0, 10))
def test_matrix_product_sequence_0beta(mats, prod_len, shift):
    num_mats = len(mats)
    assume(num_mats % prod_len == 0)
    assume(np.all(np.isfinite(mats)))
    assume(shift < num_mats)

    indices = np.arange(num_mats)[::-1]
    indices = np.roll(indices, shift)
    expected = linalg.mdot(mats[indices])
    mats = np.ascontiguousarray(mats)
    prod_seq = linalg.matrix_product_sequence_0beta(mats, prod_len, shift)
    result = reduce(np.dot, prod_seq[::-1])

    assert_allclose(result, expected, atol=1e-20)


@given(mat_arr, st.integers(1, 16), st.shared(st.integers(0, 10)))
def test_matrix_product_sequence_beta0(mats, prod_len, shift):
    num_mats = len(mats)
    assume(num_mats % prod_len == 0)
    assume(np.all(np.isfinite(mats)))
    assume(shift < num_mats)
    mats = np.ascontiguousarray(mats)
    indices = np.arange(num_mats)
    indices = np.roll(indices, -shift)
    expected = linalg.mdot(mats[indices])

    prod_seq = linalg.matrix_product_sequence_beta0(mats, prod_len, shift)
    result = reduce(np.dot, prod_seq[::-1])

    assert_allclose(result, expected, atol=1e-20)
