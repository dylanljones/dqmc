# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import logging

logger = logging.getLogger("dqmc")

# Logging format
frmt = "[%(asctime)s] %(name)s:%(levelname)-8s - %(message)s"
formatter = logging.Formatter(frmt, datefmt="%H:%M:%S")

# Set up console logger
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)

# Set up file logger
# fh = logging.FileHandler("dqmc.log", mode="w")
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# logger.addHandler(fh)

# Set logging level
logger.setLevel(logging.WARNING)
logging.root.setLevel(logging.NOTSET)
