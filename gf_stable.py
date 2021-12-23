# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
import scipy.linalg as la
from dqmc.parser import Parameters
from dqmc.stabilize import asvqrd_prod
from dqmc import hubbard_hypercube, dqmc


def qrp(arr):
    # Compute QR decomposition
    q, r, pvec = la.qr(arr, mode="full", pivoting=True)
    # Build permutation matrix as result of the column pivoting
    p = np.zeros_like(q)
    p[np.arange(len(arr)), pvec] = 1.
    return q, r, p


def matrix_product_sequence(mats, prod_len):
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


def stable_1pmatprod_inv():
    pass


def init_timestep_matrices(expk, nu, config, prod_len):
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(expk, nu, config)
    bprod_up = matrix_product_sequence(bmats_up, prod_len)
    bprod_dn = matrix_product_sequence(bmats_dn, prod_len)
    return bprod_up, bprod_dn


def main():
    prod_len = 8
    p = Parameters(5, u=4, eps=0, t=1, mu=2, dt=0.05, num_timesteps=40,
                   num_equil=100, num_sampl=100, num_recomp=1)
    # p = parse("sample_chain.txt")
    model = hubbard_hypercube(p.shape, p.u, p.eps, p.t, p.mu, p.beta)

    expk, nu, config = dqmc.init_qmc(model, p.num_timesteps)
    bmats_up, bmats_dn = dqmc.compute_timestep_mats(expk, nu, config)

    bprod_up = matrix_product_sequence(bmats_up, prod_len)
    bprod_dn = matrix_product_sequence(bmats_dn, prod_len)

    gf_up, gf_dn = asvqrd_prod(bprod_up), asvqrd_prod(bprod_dn)

    order = np.arange(p.num_timesteps)[::-1].astype(np.int64)
    gf_up_ref, gf_dn_ref = dqmc.compute_greens(bmats_up, bmats_dn, order)

    print(np.allclose(gf_up, gf_up_ref))
    print(np.allclose(gf_dn, gf_dn_ref))
    print(np.abs(gf_up - gf_up_ref) / gf_up_ref)


if __name__ == "__main__":
    main()
