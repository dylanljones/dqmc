# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from scipy import linalg as la
from abc import ABC, abstractmethod


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
