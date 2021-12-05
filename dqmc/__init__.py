# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from .logging import logger
from .model import HubbardModel, hubbard_hypercube
from .dqmc import (
    UP,
    DN,
    init_qmc,
    compute_timestep_mats,
    update_timestep_mats,
    compute_m_matrices,
    compute_greens,
    iteration_det,
    iteration_fast,
)
from .simulator import BaseDQMC, DQMC

from . import _version
__version__ = _version.get_versions()['version']
