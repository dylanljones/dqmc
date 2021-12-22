# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from scipy import linalg as la


def mdot(*arrays):
    """Computes the dot product of two or more arrays in a single function call.

    Parameters
    ----------
    arrays : sequence of array_like
        Multiple arrays or a sequence of arrays.
    Returns
    -------
    out : np.ndarray
        Returns the dot product of the supplied arrays.
    """
    if len(arrays) == 1:
        arrays = arrays[0]
    out = arrays[0]
    for mat in arrays[1:]:
        out = np.dot(out, mat)
    return out


def udr(mat):
    """Compute UDR decomposition of a matrix."""
    u, r = np.linalg.qr(mat)
    d = np.diag(r)
    r = r / d[:, np.newaxis]
    d = np.diag(d)
    return u, d, r


def reconstruct_udr(u, d, r):
    """Reconstruct a matrix from an UDR decomposition."""
    return np.dot(u, np.dot(d, r))


def svd(mat):
    u, s_diag, vh = np.linalg.svd(mat, full_matrices=True)
    s = np.diag(s_diag)
    return u, s, vh


def reconstruct_svd(u, s, vh):
    return np.dot(u, np.dot(s, vh))


def mdot_decomp(mats, chunk_size=1):
    """Calculates `B_M B_M-1 ... B_1` and stabilizes the matrix products.

    Assumes that the input `B`-matrices are ordered as `[B_1, B_2, ..., B_M]`.
    The matrix product is stabilized by intermediate matrix decompositions.

    Parameters
    ----------
    mats : float or complex (L, N, N) np.ndarray
        An array of the `B`-matrices.
    chunk_size : int, optional
        The number of matrices to multiply before applying an intermediate
        matrix decomposition.

    Returns
    -------
    u : float or complex (N, N) np.ndarray
        Unitary matrix having left singular vectors as columns.
    d : float or complex (N, N) np.ndarray
        The singular values, sorted in non-increasing order.
    t : float or complex (N, N) np.ndarray
        Unitary matrix having right singular vectors as rows.
    """
    num_mats = mats.shape[0]

    # Split into chunks of size `chunk_size` and save
    # matrix products in the stack
    indices = np.arange(num_mats)
    num_chunks = num_mats // chunk_size
    chunks = np.array_split(indices, num_chunks)
    stack = [mdot(mats[chunk]) for chunk in chunks]

    # First matrix-product to initialize U_1, D_1, T_1.
    u, d, t = udr(stack[0])
    for b_prod in stack[1:]:
        # Compute X = ((∏B U_i) D_i)
        tmp = np.dot(np.dot(b_prod, u), d)
        # Use X to compute U_{i+1}, D_{i+1}, T
        u, d, t_new = udr(tmp)
        # Compute T_{i+1} = T T_i
        t = np.dot(t_new, t)

    return reconstruct_udr(u, d, t)


def compute_greens_stable(bmats_up, bmats_dn, order):
    order_rev = order[::-1]
    eye = np.eye(bmats_up[0].shape[0], dtype=np.float64)
    m_up = eye + mdot_decomp(bmats_up[order_rev])
    m_dn = eye + mdot_decomp(bmats_dn[order_rev])
    gf_up = la.inv(m_up)
    gf_dn = la.inv(m_dn)
    return np.ascontiguousarray(gf_up), np.ascontiguousarray(gf_dn)


# =========================================================================
# UDR decomposition
# =========================================================================


def qrp(arr):
    # Compute QR decomposition
    q, r, pvec = la.qr(arr, mode="full", pivoting=True)
    # Build permutation matrix as result of the column pivoting
    p = np.zeros_like(q)
    p[np.arange(len(arr)), pvec] = 1.
    return q, r, p


def column_stratified_matrix_prod(matrices):
    # Array of matrices T_i
    tmats = np.zeros_like(matrices)

    # Compute QR decomposition with pivoting
    q, r, p = qrp(matrices[0])
    # Compute diagonal matrix D = diag(R)
    d = np.diag(np.diag(r))
    # Compute T = D^{-1} R P
    t = np.dot(np.linalg.inv(d), np.dot(r, p))
    tmats[0] = t

    for j in range(1, len(matrices)):
        tmp = np.dot(np.dot(matrices[j], q), d)
        q, r, p = qrp(tmp)
        # Compute diagonal matrix D = diag(R)
        d = np.diag(np.diag(r))
        # Compute T = D^{-1} R P
        t = np.dot(np.linalg.inv(d), np.dot(r, p))
        tmats[j] = t

    return np.dot(np.dot(q, d), mdot(tmats[::-1]))


def asvqrd_prod(matrices):
    # Array of matrices T_i
    tmats = np.zeros_like(matrices)

    # Compute QR decomposition with pivoting
    q, r, p = qrp(matrices[0])
    # Compute diagonal matrix D = diag(R)
    d = np.diag(np.diag(r))
    # Compute T = D^{-1} R P
    t = np.dot(np.linalg.inv(d), np.dot(r, p))
    tmats[0] = t

    for j in range(1, len(matrices)):
        tmp = np.dot(np.dot(matrices[j], q), d)
        q, r, p = qrp(tmp)
        # Compute diagonal matrix D = diag(R)
        d = np.diag(np.diag(r))
        # Compute T = D^{-1} R P
        t = np.dot(np.linalg.inv(d), np.dot(r, p))
        tmats[j] = t

    # Compute product of T = T_L ... T_2 T_1
    t = mdot(tmats[::-1])

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
    # calculate (D_b^{-1} Q^T + D_s T)^{-1}  (D_b^{-1} Q^T)
    rec = np.dot(la.inv(np.dot(db_inv, qt) + np.dot(ds, t)), np.dot(db_inv, qt))
    return rec
