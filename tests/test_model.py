# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from numpy.testing import assert_equal
from hypothesis import given, settings, strategies as st
from dqmc import hubbard_hypercube

settings.load_profile("dqmc")


def tb_hamiltonian_chain(num_sites, eps, mu, hop, periodic=True):
    ham = (eps - mu) * np.eye(num_sites)
    np.fill_diagonal(ham[1:, :], -hop)
    np.fill_diagonal(ham[:, 1:], -hop)
    if periodic:
        ham[0, -1] = ham[-1, 0] = -hop
    return ham


def tb_hamiltonian_square(size, eps, mu, hop, periodic=True):
    num_sites = size * size
    eye = np.eye(size)
    ham_hop_1d = np.zeros((size, size))
    np.fill_diagonal(ham_hop_1d[1:, :], -hop)
    np.fill_diagonal(ham_hop_1d[:, 1:], -hop)
    if periodic:
        ham_hop_1d[0, -1] = ham_hop_1d[-1, 0] = -hop

    ham_hop = np.kron(ham_hop_1d, eye) + np.kron(eye, ham_hop_1d)
    ham = (eps - mu) * np.eye(num_sites) + ham_hop
    return ham


@given(st.integers(5, 10),
       st.floats(0, 5),
       st.floats(0, 5),
       st.floats(0, 1),
       st.booleans())
def test_tb_hamiltonian_1d(num_sites, eps, mu, hop, periodic):
    model = hubbard_hypercube(num_sites, eps=eps, mu=mu, hop=hop, periodic=periodic)
    ham = model.hamiltonian_kinetic()
    expected = tb_hamiltonian_chain(num_sites, eps, mu, hop, periodic)
    assert_equal(expected, ham)


@given(st.integers(5, 10),
       st.floats(0, 5),
       st.floats(0, 5),
       st.floats(0, 1),
       st.booleans())
def test_tb_hamiltonian_square(num_sites, eps, mu, hop, periodic):
    shape = (num_sites, num_sites)
    model = hubbard_hypercube(shape, eps=eps, mu=mu, hop=hop, periodic=periodic)
    ham = model.hamiltonian_kinetic()
    expected = tb_hamiltonian_square(num_sites, eps, mu, hop, periodic)
    assert_equal(expected, ham)
