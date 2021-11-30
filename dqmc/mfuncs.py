# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""This module contains methods to use for the measurement of observables in the QMC simulation."""

import numpy as np


def occupation(gf_up, gf_dn):
    return 1 - np.array([np.diag(gf_up), np.diag(gf_dn)])
