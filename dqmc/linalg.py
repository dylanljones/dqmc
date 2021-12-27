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
from numba import njit, float64, int32, int64
from numba import types as nt

info_t = nt.Array(int32, 0, "C")


_PTR = ctypes.POINTER

_dble = ctypes.c_double
_char = ctypes.c_char
_int = ctypes.c_int

_ptr_dble = _PTR(_dble)
_ptr_char = _PTR(_char)
_ptr_int = _PTR(_int)

# dger(M, N, ALPHA, X, INCX, Y, INCY, A, LDA)
_dger_addr = get_cython_function_address("scipy.linalg.cython_blas", "dger")
_dger_functype = ctypes.CFUNCTYPE(None,       # Return Value
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


# dgeqp3(M, N, A, LDA, JPVT, TAU, WORK, LWORK, INFO)
_dgeqp3_addr = get_cython_function_address("scipy.linalg.cython_lapack", "dgeqp3")
_dgeqp3_functype = ctypes.CFUNCTYPE(None,       # Return Value
                                    _ptr_int,   # M
                                    _ptr_int,   # N
                                    _ptr_dble,  # A
                                    _ptr_int,   # LDA
                                    _ptr_int,   # JPVT
                                    _ptr_dble,  # TAU
                                    _ptr_dble,  # WORK
                                    _ptr_int,   # LWORK
                                    _ptr_int    # INFO
                                    )
_dgeqp3_fn = _dgeqp3_functype(_dgeqp3_addr)


# dorgqr(M, N, K, A, LDA, TAU, WORK, LWORK, INFO)
_dorgqr_addr = get_cython_function_address("scipy.linalg.cython_lapack", "dorgqr")
_dorgqr_functype = ctypes.CFUNCTYPE(None,       # Return Value
                                    _ptr_int,   # M
                                    _ptr_int,   # N
                                    _ptr_int,   # K
                                    _ptr_dble,  # A
                                    _ptr_int,   # LDA
                                    _ptr_dble,  # TAU
                                    _ptr_dble,  # WORK
                                    _ptr_int,   # LWORK
                                    _ptr_int    # INFO
                                    )
_dorgqr_fn = _dorgqr_functype(_dorgqr_addr)


@njit(
    nt.Tuple((float64[:, ::1], int32[::1], float64[::1],
              float64[::1], info_t))(float64[:, :], int64),
    nogil=True, cache=True
)
def lapack_dgeqp3(a, lwork):
    """Computes a QR factorization with column pivoting of a matrix using Level 3 BLAS.

    Calls the LAPACK subroutine `dgeqp3(M, N, A, LDA, JPVT, TAU, WORK, LWORK, INFO)`.

    Parameters
    ----------
    a : (M, N) np.ndarray
        The input matrix to compute the QR decomposition.
    lwork : int
    Returns
    -------
    qr : (M, N) np.ndarray
        The upper triangle of the array contains the `min(M,N)`-by-`N` upper
        trapezoidal matrix `R`; the elements below the diagonal, together with
        the array `tau`, represent the orthogonal matrix `Q` as a product of
        `min(M,N)` elementary reflectors.
    jpvt : (N,) np.ndarray
        If `JPVT(j)=K`, then the j-th column of `A*P` was the the k-th column of `A`.
    tau : (K,) np.ndarray
        The scalar factors of the elementary reflectors, where `K = min(M, N)`.
    work : (L, ) np.ndarray
        The work array used internally.
    info : int
        Is equal to `0` if successful exit, < 0: if INFO = -i, the i-th argument
        had an illegal value.

    Notes
    -----
    Computes the QR factorization with column pivoting of a matrix A
    ..math::
        A * P = Q * R

    The matrix Q is represented as a product of elementary reflectors
    ..math::
        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form
     ..math::
        H(i) = I - tau * v * v**T

    where tau is a real scalar, and v is a real/complex vector
    with v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in
    A(i+1:m,i), and tau in TAU(i).
    """
    _m, _n = a.shape

    m = np.array(_m, dtype=np.int32)
    n = np.array(_n, dtype=np.int32)
    qr = np.ascontiguousarray(a)
    lda = np.array(_m, dtype=np.int32)
    jpvt = np.zeros(_n, dtype=np.int32)
    tau = np.zeros(min(_m, _n), dtype=np.float64)
    info = np.array(0, dtype=np.int32)

    if lwork == -1:
        work = np.zeros(1, dtype=np.float64)
        _lwork = np.array(-1, dtype=np.int32)

        # First call with LWORK=-1 to compute optimal size of WORK array
        _dgeqp3_fn(m.ctypes,
                   n.ctypes,
                   qr.view(np.float64).ctypes,
                   lda.ctypes,
                   jpvt.view(np.int32).ctypes,
                   tau.view(np.float64).ctypes,
                   work.view(np.float64).ctypes,
                   _lwork.ctypes,
                   info.ctypes)
        # Optimal LWORK is stored in first element of WORK
        lwork = int(work[0])

    _lwork = np.array(lwork, dtype=np.int32)
    work = np.zeros(lwork, dtype=np.float64)

    # Actually compute QR decomposition
    _dgeqp3_fn(m.ctypes,
               n.ctypes,
               qr.view(np.float64).ctypes,
               lda.ctypes,
               jpvt.view(np.int32).ctypes,
               tau.view(np.float64).ctypes,
               work.view(np.float64).ctypes,
               _lwork.ctypes,
               info.ctypes)
    # Python indices start at 0, Fortran at 1
    # jpvt -= 1

    return qr, jpvt, tau, work, info


@njit(
    nt.Tuple((float64[:, ::1], float64[::1], info_t))(float64[:, :], float64[:], int64),
    nogil=True, cache=True
)
def lapack_dorgqr(qr, tau, lwork):
    """Generates the matrix Q with from the result of a QR decomposition.

     Calls the LAPACK subroutine `dorgqr(M, N, K, A, LDA, TAU, WORK, LWORK, INFO)`.

    Returns
    -------

    Notes
    -----
    Generates an M-by-N real matrix Q with orthonormal columns,
    which is defined as the first N columns of a product of K elementary
    reflectors H of order M.
    ..math::
        Q  =  H(1) H(2) . . . H(k).
    """
    _m, _n = qr.shape

    m = np.array(_m, dtype=np.int32)
    n = np.array(_n, dtype=np.int32)
    k = np.array(len(tau), dtype=np.int32)
    q = np.ascontiguousarray(qr)
    lda = np.array(_m, dtype=np.int32)
    info = np.array(0, dtype=np.int32)

    if lwork == -1:
        work = np.zeros(1, dtype=np.float64)
        _lwork = np.array(-1, dtype=np.int32)

        # First call with LWORK=-1 to compute optimal size of WORK array
        _dorgqr_fn(m.ctypes,
                   n.ctypes,
                   k.ctypes,
                   q.view(np.float64).ctypes,
                   lda.ctypes,
                   tau.view(np.float64).ctypes,
                   work.view(np.float64).ctypes,
                   _lwork.ctypes,
                   info.ctypes)

        # Optimal LWORK is stored in first element of WORK
        lwork = int(work[0])

    _lwork = np.array(lwork, dtype=np.int32)
    work = np.zeros(lwork, dtype=np.float64)

    # Actually compute Q-matrix
    _dorgqr_fn(m.ctypes,
               n.ctypes,
               k.ctypes,
               q.view(np.float64).ctypes,
               lda.ctypes,
               tau.view(np.float64).ctypes,
               work.view(np.float64).ctypes,
               _lwork.ctypes,
               info.ctypes)

    return q, work, info


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


def qrp2(arr):
    # Compute QR decomposition
    q, r, jpvt = la.qr(arr, mode="full", pivoting=True)
    # Build permutation matrix as result of the column pivoting
    n = len(jpvt)
    p = np.zeros((n, n), dtype=np.int64)
    p[jpvt, np.arange(n)] = 1
    return q, r, p


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


def asvqrd_prod_0beta(matrices, t=0, prod_len=1):
    """Computes the stabilized inverse of a matrix product plus the identity.

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
    res : (N, M) np.ndarray
        The result of the matrix product and inversion.

    Notes
    -----
    Uses the stratification methods with pre-pivoting to compute
    ..math::
        G = (I + B_{L-l-1} B_{L-l-2} ... B_{l+1}, B_{l})^{-1}
    """
    # Pre-compute explicit matrix products
    mats = matrix_product_sequence_0beta(matrices, prod_len, t)

    # Compute QR decomposition with pivoting
    q, r, p = qrp(mats[0])
    # Compute diagonal matrix D = diag(R)
    d = np.diag(np.diag(r))
    # Compute T = D^{-1} R P
    t = np.dot(np.linalg.inv(d), np.dot(r, p.T))
    t_prod = t
    for j in range(1, len(mats)):
        tmp = np.dot(np.dot(mats[j], q), d)
        q, r, p = qrp(tmp)
        # Compute diagonal matrix D = diag(R)
        d = np.diag(np.diag(r))
        # Compute T = D^{-1} R P
        t = np.dot(np.linalg.inv(d), np.dot(r, p.T))
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


def asvqrd_prod_beta0(matrices, prod_len, t):
    """Computes the stabilized inverse of a matrix product plus the identity.

    Parameters
    ----------
    matrices : (L, N, M) np.ndarray
        The input matrices to compute the matrix product and inverse. Expected
        to be in normal order :math:'B_0, B_1, ..., B_{L-1}'.
    prod_len : int
        The number of matrices in each product. Has to be a multiple of
        the total number of matrices `L`.
    t : int
        The current time slice index `l`.

    Returns
    -------
    res : (N, M) np.ndarray
        The result of the matrix product and inversion.

    Notes
    -----
    Uses the stratification methods with pre-pivoting to compute
    ..math::
        G = (I + B_l B_{l+1} ... B_{L-l-2}, B_{L-l-1})^{-1}
    """
    # Pre-compute explicit matrix products
    mats = matrix_product_sequence_beta0(matrices, prod_len, t)

    # Compute QR decomposition with pivoting
    q, r, p = qrp(mats[0])
    # Compute diagonal matrix D = diag(R)
    d = np.diag(np.diag(r))
    # Compute T = D^{-1} R P
    t = np.dot(np.linalg.inv(d), np.dot(r, p.T))
    t_prod = t
    for j in range(1, len(mats)):
        tmp = np.dot(np.dot(mats[j], q), d)
        q, r, p = qrp(tmp)
        # Compute diagonal matrix D = diag(R)
        d = np.diag(np.diag(r))
        # Compute T = D^{-1} R P
        t = np.dot(np.linalg.inv(d), np.dot(r, p.T))
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
