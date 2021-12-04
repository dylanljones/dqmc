# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""This module contains methods for measuring observables in the QMC simulation."""

import numpy as np


def occupation(gf_up, gf_dn):
    return 1 - np.array([np.diag(gf_up), np.diag(gf_dn)])


def spin_z(gf_up, gf_dn):
    n_up = 1 - np.diag(gf_up)
    n_dn = 1 - np.diag(gf_dn)
    return n_up - n_dn


def mz_moment(gf_up, gf_dn):
    n_up = 1 - np.diag(gf_up)
    n_dn = 1 - np.diag(gf_dn)
    return n_up + n_dn - 2 * n_up * n_dn
