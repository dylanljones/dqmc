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

    def __init__(self, num_sites, u=0.0, eps=0.0, hop=1.0, mu=0.):
        self.u = u
        self.hop = hop
        self.eps = eps
        self.mu = mu
        self.num_sites = num_sites

    @classmethod
    def half_filled(cls, num_sites, u=0.0, eps=0.0, hop=1.0):
        mu = u/4 - eps
        return cls(num_sites, u, eps, hop, mu)

    def hamiltonian_kinetic(self):
        """Builds tridiagonal kinetic Hamiltonian for 1D Hubbard chain."""
        diag = (self.eps - self.mu) * np.ones(self.num_sites)
        offdiag = - self.hop * np.ones(self.num_sites - 1)
        arrs = [offdiag, diag, offdiag]
        offset = [-1, 0, +1]
        return diags(arrs, offset).toarray()
