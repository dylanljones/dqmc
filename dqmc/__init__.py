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
from .simulator import DQMC, run_dqmc, parse, Parameters
from .mp import map_params, run_dqmc_parallel

from . import _version
__version__ = _version.get_versions()['version']
