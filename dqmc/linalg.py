# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
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


def decompose_qrp(a):
    """Performs a QRP decomposition (with column pivoting) of a square matrix `A`.

    The QR decomposition with column pivoting is defined as:
    .. math:
        A = Q R P

    where `P` is the permutation matrix as a result of the column pivoting.

    Parameters
    ----------
    a : (N, N) np.ndarray
        The input matrix `A` to decompose.

    Returns
    -------
    q : (N, N) np.ndarray
        The orthogonal matrix `Q`.
    r : (N, N) np.ndarray
        The upper triangular matrix `R`.
    jpvt : (N) np.ndarray
        The column pivoting indices which define the permutation matrix `P`.

    Notes
    -----
    Instead of the perumtation matrix the indices of the columns as a result of the
    pivoting are returned to reduce the memory used. To restore the original order
    of the input matrix columns these can be simply used as indices.
    """
    assert a.shape[0] == a.shape[1]

    # Call DGEQP3 to compute QR and tau matrices and the collumn pivot indices
    qr, jpvt, tau, work, info = la.lapack.dgeqp3(a)

    n = qr.shape[0]
    # Call DORGQR to construct Q and R matrix from QR
    r = np.triu(qr[:n, :])
    q, work, info = la.lapack.dorgqr(qr, tau)

    # Sort pivoting indices for column pivoting
    return q, r, np.argsort(jpvt)


def reconstruct_qrp(q, r, jpvt):
    """Reconstructs the original matrix `A` from a QRP decomposition.

    Parameters
    ----------
    q : (N, N) np.ndarray
        The orthogonal matrix `Q`.
    r : (N, N) np.ndarray
        The upper triangular matrix `R`.
    jpvt : (N) np.ndarray
        The column pivoting indices which define the permutation matrix `P`.

    Returns
    -------
    a : (N, N) np.ndarray
        The reconstructed matrix `A`.
    """
    return np.dot(q, r)[:, jpvt]


def decompose_udt(a):
    r"""Performs a UDT decomposition of a square matrix `A`.

    The UDT decomposition can be constructed from the result of a QRP decomposition.
    Starting from
    .. math::
        A = Q R P

    The matrix `D` is given by the diagonal entries of the upper triangular matrix `R`:
    .. math::
        D = \diag(\abs(R))

    The definition of `D` makes the matrix `T` well-conditioned. It's columns are
    scaled by the inverse of `D`:
    .. math::
        T = D^{-1} R P

    Parameters
    ----------
    a : (N, N) np.ndarray
        The input matrix `A` to decompose.

    Returns
    -------
    u : (N, N) np.ndarray
        The orthogonal matrix `U`.
    d : (N) np.ndarray
        The diagonal entries of the matrix `D`.
    t : (N, N) np.ndarray
        The well-conditioned matrix `T`.

    References
    ----------
    .. [1] Z. Bai et al., "Stable solutions of linear systems involving long chain
           of matrix multiplications", Linear Algebra Appl. 435, 659-673 (2011)
    """
    q, r, jpvt = decompose_qrp(a)
    # Extract diagonal entries of R
    d = np.diag(np.abs(r))
    # Scale columns of R by elements of 1/D and apply column pivoting
    t = (r.T / d).T[:, jpvt]
    return q, d, t


def reconstruct_udt(u, d, t):
    """Reconstructs the original matrix `A` from a UDT decomposition.

    Parameters
    ----------
    u : (N, N) np.ndarray
        The orthogonal matrix `U`.
    d : (N) np.ndarray
        The diagonal entries of the matrix `D`.
    t : (N, N) np.ndarray
        The well-conditioned matrix `T`.

    Returns
    -------
    a : (N, N) np.ndarray
        The reconstructed matrix `A`.
    """
    return np.dot(u, (d * t.T).T)


def mdot_udt(mats, prod_len=1):
    num_mats = len(mats)
    assert num_mats % prod_len == 0
    num_blocks = int(num_mats / prod_len)
    mat = mdot(mats[:prod_len]) if prod_len > 1 else mats[0]
    u, d, t = decompose_udt(mat)
    for i in range(1, num_blocks):
        mat = mdot(mats[i * prod_len:(i + 1) * prod_len]) if prod_len > 1 else mats[i]
        u_r, d_r, t_r = decompose_udt(mat)
        # Compute intermediate UDT decomposition of D (T U_R) D_R
        tmp = (d * np.dot(t, u_r).T).T * d_r
        u_c, d, t_c = decompose_udt(tmp)
        # Compute new U and T
        u = np.dot(u, u_c)
        t = np.dot(t_c, t_r)
    return u, d, t


def inv_one_plus_mdot(mats):
    return la.inv(np.eye(mats.shape[1]) + mdot(mats))


def inv_one_plus_mdot_udt(mats, prod_len=1):
    u, dm, t = mdot_udt(mats, prod_len)
    # Constrcut D_m = min(D, 1) and D_p = max(D, 1)
    dm = np.array(dm)
    dp = np.copy(dm)
    dm[dm > 1] = 1
    dp[dp < 1] = 1
    tl = t
    u, d, t = decompose_udt(np.dot(la.inv(t), np.diag(1 / dp)) + np.dot(u, np.diag(dm)))
    u, dr, tr = decompose_udt(np.dot(np.diag(dp), la.inv(reconstruct_udt(u, d, t))))
    ur = np.dot(la.inv(tl), u)
    return ur, dr, tr


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
    .. math::
        B_0, B_1, ..., B_{K-1} -> [B_{L-1}• ... •B_0], ..., [B_{K-1}• ... •B_{K-L-1}]

    where L is the number of matrices. If a shift is given, the indices
    are shifted in reverse order. For example, if a shift of 1 is given:
    .. math::
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

    seq_indices = np.split(indices, num_seqs)
    prod_seq = np.zeros((num_seqs, n, m), dtype=np.float64)
    for s, idx in enumerate(seq_indices):
        prod = mats[idx[0]]
        for j in idx[1:]:
            prod = np.dot(mats[j], prod)
        prod_seq[s] = prod
    return prod_seq


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
        return mats[indices[::-1]]

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
