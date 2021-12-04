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
from scipy.sparse import csr_matrix
from lattpy import Lattice


class HubbardModel(Lattice):
    def __init__(self, vectors, u=2.0, eps=0.0, hop=1.0, mu=0.0, beta=5.0):
        super().__init__(vectors)
        self.u = u
        self.hop = hop
        self.eps = eps
        self.mu = mu
        self.beta = beta

    def set_beta(self, beta):
        self.beta = beta

    def set_temperature(self, temp):
        self.beta = 1 / temp

    @classmethod
    def half_filled(cls, num_sites, u=0.0, eps=0.0, hop=1.0, beta=1.0):
        mu = u / 2 - eps
        return cls(num_sites, u, eps, hop, mu, beta)

    def hamiltonian_kinetic(self):
        """Builds tridiagonal kinetic Hamiltonian for 1D Hubbard chain."""
        hop = -self.hop
        onsite = self.eps - self.mu

        dmap = self.data.map()
        data = np.zeros(dmap.size, dtype=np.float64)
        data[dmap.onsite()] = onsite
        data[dmap.hopping()] = hop
        return csr_matrix((data, dmap.indices)).toarray()


def hubbard_hypercube(shape, u=0.0, eps=0.0, hop=1.0, mu=0.0, beta=0.0, periodic=None):
    dim = 1 if isinstance(shape, int) else len(shape)
    if isinstance(periodic, bool) and periodic:
        periodic = np.arange(dim)

    model = HubbardModel(np.eye(dim), u, eps, hop, mu, beta)
    model.add_atom()
    model.add_connections(1)
    model.build(shape, relative=True, periodic=periodic)
    return model
