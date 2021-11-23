# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from scipy.sparse import diags


class HubbardModel:

    def __init__(self, num_sites, u=2.0, eps=0.0, hop=1.0, mu=0., beta=5.):
        self.u = u
        self.hop = hop
        self.eps = eps
        self.mu = mu
        self.num_sites = num_sites
        self.beta = beta

    @classmethod
    def half_filled(cls, num_sites, u=0.0, eps=0.0, hop=1.0, beta=1.):
        mu = u/2 - eps
        return cls(num_sites, u, eps, hop, mu, beta)

    def hamiltonian_kinetic(self, periodic=True):
        """Builds tridiagonal kinetic Hamiltonian for 1D Hubbard chain."""
        hop = -self.hop
        diag = (self.eps - self.mu) * np.ones(self.num_sites)
        offdiag = hop * np.ones(self.num_sites - 1)
        arrs = [offdiag, diag, offdiag]
        offset = [-1, 0, +1]
        ham = diags(arrs, offset).toarray()
        if periodic:
            ham[0, -1] = hop
            ham[-1, 0] = hop
        return ham
