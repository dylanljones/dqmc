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

__all__ = ["HubbardModel", "hubbard_hypercube"]


class HubbardModel(Lattice):
    """Hubbard lattice model.

    Parameters
    ----------
    vectors : (D, D) array_like or float
        The vectors that span the basis of the `D`-dimensional lattice.
    u : float, optional
        The onsite interaction energy `U`. The default value is `4.0`.
    eps : float, optional
        The onsite energy `ε`. The default value is `0.0`.
    hop : float, optional
        The absolut value of the hopping parameter `t`. The default value is `1.0`.
        Note that the Hamiltonian is built using the negative of the hopping parameter.
    mu : float, optional
        The chemical chemical potential `μ`. The default is `0`.
        The chemical potential is subtracted from the on-site energy.
    beta : float, optional
        The inverse of the temperature `β=1/T`
    """

    def __init__(self, vectors, u=2.0, eps=0.0, hop=1.0, mu=0.0, beta=5.0):
        super().__init__(vectors)
        self.u = u
        self.hop = hop
        self.eps = eps
        self.mu = mu
        self.beta = beta

    @classmethod
    def half_filled(cls, num_sites, u=0.0, eps=0.0, hop=1.0, beta=1.0):
        """Creates a Hubbard model at half filling with `μ = U/2`."""
        mu = u / 2 - eps
        return cls(num_sites, u, eps, hop, mu, beta)

    def set_beta(self, beta):
        """Set's the inverse temperature `β=1/T`."""
        self.beta = beta

    def set_temperature(self, temp):
        """Set's the temperature `T` by computing the inverse temperature `β=1/T`."""
        self.beta = 1 / temp

    def hamiltonian_kinetic(self):
        r"""Builds the kinetic (tight-binding) Hamiltonian for the Hubbard model.

        The tight binding hamiltonian includes the hopping `t`, the on-site energy `ε`
        and the chemical potential `μ`:
        .. math::

            H = - \mathtt{t} Σ_{i,j} c^†_i c_j + Σ_i (\mathtt{ε} - \mathtt{μ}) c^†_i c_i

        Retrurns
        --------
        ham : (N, N) np.ndarray
            The Hamiltonian matrix, where `N` is the number of lattice sites.
        """
        hop = -self.hop
        onsite = self.eps - self.mu

        dmap = self.data.map()
        data = np.zeros(dmap.size, dtype=np.float64)
        data[dmap.onsite()] = onsite
        data[dmap.hopping()] = hop
        return csr_matrix((data, dmap.indices)).toarray()


def hubbard_hypercube(shape, u=0.0, eps=0.0, hop=1.0, mu=0.0, beta=0.0, periodic=None):
    """Construct a `d`-dimensional Hubbard model.

    Parameters
    ----------
    shape : array_like or int
        The shape of the model. If a sequence is passed the length determines
        the dimensionality of the lattice. In case of an integer a 1D lattice
        is constructed.
    u : float, optional
        The onsite interaction energy `U`. The default value is `4.0`.
    eps : float, optional
        The onsite energy `ε`. The default value is `0.0`.
    hop : float, optional
        The absolut value of the hopping parameter `t`. The default value is `1.0`.
        Note that the Hamiltonian is built using the negative of the hopping parameter.
    mu : float, optional
        The chemical chemical potential `μ`. The default is `0`.
        The chemical potential is subtracted from the on-site energy.
    beta : float, optional
        The inverse of the temperature `β=1/T`
    periodic : sequence or integer or bool, optional
        Periodic boundary conditions. An integer or a sequence of integers is
        interpreted as the periodic axes to set, a boolean enables or disables
        all axes to be periodic.

    Returns
    -------
    model : HubbardModel
        The fully initializes hyper-rectanlge Hubbard model.
    """
    dim = 1 if isinstance(shape, int) else len(shape)
    if isinstance(periodic, bool):
        if periodic:
            periodic = np.arange(dim)
        else:
            periodic = None
    model = HubbardModel(np.eye(dim), u, eps, hop, mu, beta)
    model.add_atom()
    model.add_connections(1)
    model.build(shape, relative=True, periodic=periodic)
    return model
