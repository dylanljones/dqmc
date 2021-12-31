# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import ctypes
import numpy as np
from scipy import linalg as la
from numba.extending import get_cython_function_address
from numba import njit, float64, int64


_dble = ctypes.POINTER(ctypes.c_double)
_int = ctypes.POINTER(ctypes.c_int)

# dger(M, N, ALPHA, X, INCX, Y, INCY, A, LDA)
_ft = ctypes.CFUNCTYPE(None, _int, _int, _dble, _dble, _int, _dble, _int, _dble, _int)
_dger_fn = _ft(get_cython_function_address("scipy.linalg.cython_blas", "dger"))


@njit((float64, float64[:], float64[:], float64[:, :]), nogil=True, cache=True)
def blas_dger(alpha, x, y, a):
    """Performs the rank 1 operation .math:`A = α x•y^T + A` via a BLAS call.

    Parameters
    ----------
    alpha : float
        A scalar factor of the rank1 update.
    x : (M, ) np.ndarray
        An M element collumn vector.
    y : (N, ) np.ndarray
        An N element row vector.
    a : (M, N) np.ndarray
        The MxN matrix to be updated.
    """
    _m, _n = a.shape

    alpha = np.array(alpha, dtype=np.float64)
    m = np.array(_m, dtype=np.int32)
    n = np.array(_n, dtype=np.int32)
    incx = np.array(1, np.int32)
    incy = np.array(1, np.int32)
    lda = np.array(_m, np.int32)

    _dger_fn(m.ctypes,
             n.ctypes,
             alpha.ctypes,
             y.view(np.float64).ctypes,
             incy.ctypes,
             x.view(np.float64).ctypes,
             incx.ctypes,
             a.view(np.float64).ctypes,
             lda.ctypes)


@njit(
    (float64, float64[:], float64[:], float64[:, :]),
    nogil=True, cache=True, fastmath=True
)
def numpy_dger(alpha, x, y, a):
    """Performs the rank 1 operation .math:`A = α x•y^T + A` via numpy methods.

    Parameters
    ----------
    alpha : float
        A scalar factor of the rank1 update.
    x : (M, ) np.ndarray
        An M element collumn vector.
    y : (N, ) np.ndarray
        An N element row vector.
    a : (M, N) np.ndarray
        The MxN matrix to be updated.
    """
    a[:, :] = alpha * np.outer(x, y) + a


@njit(float64[:, :](float64[:, :, ::1]), fastmath=True, nogil=True, cache=True)
def mdot(mats):
    r"""Computes the dot-product of multiple matrices.

    Parameters
    ----------
    mats : (L, N, N) np.ndarray
        The input matrices in the order they are multiplied.

    Returns
    -------
    prod : (N, N) np.ndarray
        The dot-product of multiple matrices.
    """
    prod = mats[0]
    for mat in mats[1:]:
        prod = np.dot(prod, mat)
    return prod


def qrp(a):
    # Call DGEQP3 to compute QR and tau matrices and the collumn pivot indices
    qr, jpvt, tau, work, info = la.lapack.dgeqp3(a)

    m, n = qr.shape
    # Call DORGQR to construct Q and R matrix from QR
    r = np.triu(qr[:n, :])
    if m < n:
        q, work, info = la.lapack.dorgqr(qr[:, :m], tau)
    else:
        qqr = np.empty((m, n), dtype=qr.dtype)
        qqr[:, :n] = qr
        q, work, info = la.lapack.dorgqr(qqr, tau)

    # Build permutation matrix as result of the column pivoting
    jpvt -= 1  # Fortran indices start with 1
    n = len(jpvt)
    p = np.zeros((n, n), dtype=np.int64)
    p[jpvt, np.arange(n)] = 1

    return q, r, p


def reconstruct_qrp(q, r, p):
    return np.dot(np.dot(q, r), p.T)


@njit(
    float64[:, :, :](float64[:, :, ::1], int64, int64),
    fastmath=True, nogil=True, cache=True
)
def matrix_product_sequence_0beta(mats, prod_len, shift):
    """Returns a sequence of matrix products in reverse order of length `L`.

    Parameters
    ----------
    mats : (K, N, M) np.ndarray
        The input matrices to compute the matrix product sequence.
    prod_len : int
        The number of matrices `L` in each product. Has to be a multiple of
        the total number of matrices `K`.
    shift : int
        An integer defining the shift of indices.
    Returns
    -------
    prod_seq : (K, N, M) np.ndarray
        The matrix product sequence, where `K` is the number of matrices `L`
        devided by `prod_len`.

    Notes
    -----
    The input matrices :math:'B_1, B_2, ..., B_K' are multiplied in
    reverse order in each segment:
    ..math::
        B_0, B_1, ..., B_{K-1} -> [B_{L-1}• ... •B_0], ..., [B_{K-1}• ... •B_{K-L-1}]

    where L is the number of matrices. If a shift is given, the indices
    are shifted in reverse order. For example, if a shift of 1 is given:
        ..math::
        B_0, B_1, ..., B_{K-1} -> [B_{L-2}}• ... •B_1], ..., [B_0•B_{K-1} ... •B_{K-L}]
    """
    num_mats = len(mats)
    assert (num_mats % prod_len) == 0
    num_seqs = int(num_mats / prod_len)
    n, m = mats[0].shape
    indices = np.arange(num_mats)[::-1]
    indices = np.roll(indices, shift)
    if prod_len == 1:
        return mats[indices]

    prod_seq = np.zeros((num_seqs, n, m), dtype=np.float64)
    for i in range(num_seqs):
        i0 = i * prod_len
        i1 = i0 + prod_len
        prod_indices = indices[i0:i1]
        prod = mats[prod_indices[0]]
        for j in prod_indices[1:]:
            prod = np.dot(prod, mats[j])
        prod_seq[i] = prod
    return prod_seq[::-1]


@njit(
    float64[:, :, :](float64[:, :, ::1], int64, int64),
    fastmath=True, nogil=True, cache=True
)
def matrix_product_sequence_beta0(mats, prod_len, shift):
    """Returns a sequence of matrix products in reverse order of length `L`.

    Parameters
    ----------
    mats : (K, N, M) np.ndarray
        The input matrices to compute the matrix product sequence.
    prod_len : int
        The number of matrices `L` in each product. Has to be a multiple of
        the total number of matrices `K`.
    shift : int
        An integer defining the shift of indices.
    Returns
    -------
    prod_seq : (K, N, M) np.ndarray
        The matrix product sequence, where `K` is the number of matrices `L`
        devided by `prod_len`.

    Notes
    -----
    The input matrices :math:'B_1, B_2, ..., B_K' are multiplied in
    reverse order in each segment:
    ..math::
        B_0, B_1, ..., B_{K-1} -> [B_{L-1}• ... •B_0], ..., [B_{K-1}• ... •B_{K-L-1}]

    where L is the number of matrices. If a shift is given, the indices
    are shifted in reverse order. For example, if a shift of 1 is given:
        ..math::
        B_0, B_1, ..., B_{K-1} -> [B_{L-2}}• ... •B_1], ..., [B_0•B_{K-1} ... •B_{K-L}]
    """
    num_mats = len(mats)
    assert (num_mats % prod_len) == 0
    num_seqs = int(num_mats / prod_len)
    n, m = mats[0].shape
    indices = np.arange(num_mats)
    indices = np.roll(indices, -shift)
    if prod_len == 1:
        return mats[indices]

    prod_seq = np.zeros((num_seqs, n, m), dtype=np.float64)
    for i in range(num_seqs):
        i0 = i * prod_len
        i1 = i0 + prod_len
        prod_indices = indices[i0:i1]
        prod = mats[prod_indices[0]]
        for j in prod_indices[1:]:
            prod = np.dot(prod, mats[j])
        prod_seq[i] = prod
    return prod_seq[::-1]


def timeflow_map(mats):
    # Compute first QR decomposition with column pivoting
    q, jpvt, tau, work, info = la.lapack.dgeqp3(mats[0])
    # Extract diagonal elements of R (upper triangular matrix of Q)
    d = np.array(np.diag(q))
    d[d == 0.0] = 1.0
    # Compute T_1 = D^{-1} R P:
    # Multiply columns of R with 1/d and apply column pivoting
    t = (np.triu(q).T / d).T[:, np.argsort(jpvt)]
    for i in range(1, len(mats)):
        lwork = len(work)  # noqa
        # Multiply with Q from the right, overwriting the 'W' matrix
        w, work, info = la.lapack.dormqr("R", "N", q, tau, mats[i], lwork)
        # Scale by previous diagonal entries
        w *= d
        # Pre-pivot 'W' and perform QR-decomposition
        jpvt = np.argsort(la.norm(w, axis=0))[::-1]
        q, tau, work, info = la.lapack.dgeqrf(w[:, jpvt])
        # Extract diagonal elements of R (upper triangular matrix of Q)
        d = np.array(np.diag(q))
        d[d == 0.0] = 1.0
        # Multiply 1/d with the upper triangular R matrix
        # and multiply current T matrix with the pivoted product of the previous T's
        t = np.dot((np.triu(q).T / d).T, t[jpvt, :])

    lwork = len(work)  # noqa
    return q, d, t, tau, lwork


def timeflow_map_0beta(matrices, prod_len, t):
    """Computes the UDT decomposition for simulations in ascending time flow order.

    Parameters
    ----------
    matrices : (L, N, M) np.ndarray
        The input matrices to compute the matrix product and inverse. Expected
        to be in normal order :math:'B_0, B_1, ..., B_{L-1}'.
    t : int
        The current time slice index `l`.
    prod_len : int
        The number of matrices in each product. Has to be a multiple of
        the total number of matrices `L`.
    Returns
    -------
    q : (N, N) np.ndarray
        The unitary/orthogonal matrix Q of the QR decomposition.
    d : (N, ) np.ndarray
        The diagonal elements of the matrix R of the QR decomposition.
    t : (N, N) np.ndarray
        The product of T matrices.
    tau : (N, ) np.ndarray
        The scalar factors of the elementary reflectors of the QR decomposition.
    lwork : int
        The used size of the work array.

    References
    ----------
    .. [1] Z. Bai et al., “Stable solutions of linear systems involving long chain
           of matrix multiplications”, in Linear Algebra Appl. 435, p. 659-673 (2011)
    """
    # Pre-compute matrix product sequence
    mats = matrix_product_sequence_0beta(matrices, prod_len, t)
    # Call Scipy implementation
    return timeflow_map(mats)


def timeflow_map_beta0(matrices, prod_len, t):
    """Computes the UDT decomposition for simulations in descending time flow order.

    Parameters
    ----------
    matrices : (L, N, M) np.ndarray
        The input matrices to compute the matrix product and inverse. Expected
        to be in normal order :math:'B_0, B_1, ..., B_{L-1}'.
    t : int
        The current time slice index `l`.
    prod_len : int
        The number of matrices in each product. Has to be a multiple of
        the total number of matrices `L`.

    Returns
    -------
    q : (N, N) np.ndarray
        The unitary/orthogonal matrix Q of the QR decomposition.
    d : (N, ) np.ndarray
        The diagonal elements of the matrix R of the QR decomposition.
    t : (N, N) np.ndarray
        The product of T matrices.
    tau : (N, ) np.ndarray
        The scalar factors of the elementary reflectors of the QR decomposition.
    lwork : int
        The used size of the work array.
    """
    # Pre-compute matrix product sequence
    mats = matrix_product_sequence_beta0(matrices, prod_len, t)
    # Call Scipy implementation
    return timeflow_map(mats)
