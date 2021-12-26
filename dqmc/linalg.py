# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import ctypes
import numpy as np
from scipy import linalg as la
from numba.extending import get_cython_function_address
from numba import njit, float64, int64

_PTR = ctypes.POINTER

_dble = ctypes.c_double
_char = ctypes.c_char
_int = ctypes.c_int

_ptr_dble = _PTR(_dble)
_ptr_char = _PTR(_char)
_ptr_int = _PTR(_int)


_dger_addr = get_cython_function_address('scipy.linalg.cython_blas', 'dger')
_dger_functype = ctypes.CFUNCTYPE(None,
                                  _ptr_int,   # M
                                  _ptr_int,   # N
                                  _ptr_dble,  # ALPHA
                                  _ptr_dble,  # X
                                  _ptr_int,   # INCX
                                  _ptr_dble,  # Y
                                  _ptr_int,   # INCY
                                  _ptr_dble,  # A
                                  _ptr_int,   # LDA
                                  )
_dger_fn = _dger_functype(_dger_addr)


@njit((float64, float64[:], float64[:], float64[:, :]), nogil=True, cache=True)
def dger(alpha, x, y, a):
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
def dger_numpy(alpha, x, y, a):
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


@njit(
    float64[:, :, :](float64[:, :, ::1], int64), fastmath=True, nogil=True, cache=True
)
def matrix_product_sequence(mats, prod_len):
    """Returns a sequence of matrix products of length `prod_len`.

    Parameters
    ----------
    mats : (L, N, M) np.ndarray
        The input matrices to compute the matrix product sequence.
    prod_len : int
        The number of matrices in each product. Has to be a multiple of
        the total number of matrices `L`.

    Returns
    -------
    prod_seq : (K, N, M) np.ndarray
        The matrix product sequence, where `K` is the number of matrices `L`
        devided by `prod_len`.
    """
    num_mats = len(mats)
    assert (num_mats % prod_len) == 0
    num_seqs = int(num_mats / prod_len)
    n, m = mats[0].shape
    prod_seq = np.zeros((num_seqs, n, m), dtype=np.float64)
    for i in range(num_seqs):
        i0 = i * prod_len
        i1 = i0 + prod_len
        chunk = mats[i0:i1][::-1]
        prod = chunk[0]
        for mat in chunk[1:]:
            prod = np.dot(prod, mat)
        prod_seq[i] = prod
    return prod_seq


def qrp(arr):
    # Compute QR decomposition
    q, r, pvec = la.qr(arr, mode="full", pivoting=True)
    # Build permutation matrix as result of the column pivoting
    p = np.zeros_like(q)
    p[np.arange(len(arr)), pvec] = 1.
    return q, r, p


def asvqrd_prod(matrices, prod_len):
    """Computes the stabilized inverse of a matrix product plus the identity.

    Uses the stratification methods with pre-pivoting to compute
    ..math::
        G = (I + B_L ... B_2, B_1)^{-1}

    Parameters
    ----------
    matrices : (L, N, M) np.ndarray
        The input matrices to compute the matrix product and inverse.
    prod_len : int
        The number of matrices in each product. Has to be a multiple of
        the total number of matrices `L`.

    Returns
    -------
    res : (N, M) np.ndarray
        The result of the matrix product and inversion.
    """
    # Pre-compute explicit matrix products
    mats = matrix_product_sequence(matrices, prod_len)

    # Compute QR decomposition with pivoting
    q, r, p = qrp(mats[0])
    # Compute diagonal matrix D = diag(R)
    d = np.diag(np.diag(r))
    # Compute T = D^{-1} R P
    t = np.dot(np.linalg.inv(d), np.dot(r, p))
    t_prod = t
    for j in range(1, len(mats)):
        tmp = np.dot(np.dot(mats[j], q), d)
        q, r, p = qrp(tmp)
        # Compute diagonal matrix D = diag(R)
        d = np.diag(np.diag(r))
        # Compute T = D^{-1} R P
        t = np.dot(np.linalg.inv(d), np.dot(r, p))
        # Compute product of T = T_L ... T_2 T_1
        t_prod = np.dot(t, t_prod)

    # Compute matrices D_b and D_s, such that D_L = D_b D_s
    diag = np.diag(d)
    db = np.eye(len(d))
    ds = np.eye(len(d))
    for i in range(len(d)):
        absdiag = abs(diag[i])
        if absdiag > 1:
            db[i, i] = diag[i]
        else:
            ds[i, i] = diag[i]

    db_inv = la.inv(db)
    qt = q.T
    # calculate (D_b^{-1} Q^T + D_s T)^{-1} (D_b^{-1} Q^T)
    return np.dot(la.inv(np.dot(db_inv, qt) + np.dot(ds, t_prod)), np.dot(db_inv, qt))
