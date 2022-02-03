# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""DQMC Parameter object and helper methods."""

import logging
from typing import Union
from dataclasses import dataclass

logger = logging.getLogger("dqmc")


@dataclass
class Parameters:

    shape: Union[int, tuple]
    u: float = 0.0
    eps: float = 0.0
    t: float = 1.0
    mu: float = 0.
    dt: float = 0.01
    num_times: int = 40
    num_equil: int = 512
    num_sampl: int = 2048
    num_wraps: int = 1
    prod_len: int = 1
    sampl_recomp: int = 1
    seed: int = 0

    def copy(self, **kwargs):
        # Copy parameters
        p = Parameters(**self.__dict__)
        # Update new parameters with given kwargs
        for key, val in kwargs.items():
            setattr(p, key, val)
        return p

    @property
    def beta(self):
        return self.num_times * self.dt

    @beta.setter
    def beta(self, beta):
        self.dt = beta / self.num_times

    @property
    def temp(self):
        return 1 / (self.num_times * self.dt)

    @temp.setter
    def temp(self, temp):
        self.dt = 1 / (temp * self.num_times)


def parse(file):  # noqa: C901
    """Parses an input text file and extracts the DQMC parameters.

    Parameters
    ----------
    file : str
        The path of the input file.
    Returns
    -------
    p : Parameters
        The parsed parameters of the input file.
    """
    shape = 0
    u = 0.
    eps = 0.
    t = 0.
    mu = 0.
    dt = 0.
    beta = 0.
    temp = 0.
    num_timesteps = 0
    warm = 0
    meas = 0
    num_recomp = 0
    sampl_recomp = 1
    prod_len = 1

    logger.info("Parsing parameters from file %s...", file)
    with open(file, "r") as fh:
        text = fh.read()
    lines = text.splitlines(keepends=False)
    for line in lines:
        if "#" in line:
            text, comm = line.strip().split("#")
            line = text.strip()
        if not line:
            continue

        head, val = line.split(maxsplit=1)
        head = head.lower()
        if head == "shape":
            shape = tuple(int(x) for x in val.split(", "))
        elif head == "u":
            u = float(val)
        elif head == "eps":
            eps = float(val)
        elif head == "t":
            t = float(val)
        elif head == "mu":
            mu = float(val)
        elif head == "dt":
            dt = float(val)
        elif head == "l":
            num_timesteps = int(val)
        elif head == "nequil":
            warm = int(val)
        elif head == "nsampl":
            meas = int(val)
        elif head == "nwraps":
            num_recomp = int(val)
        elif head == "recomp":
            sampl_recomp = int(val)
        elif head == "prodlen":
            prod_len = int(val)
        elif head == "beta":
            beta = float(val)
        elif head == "temp":
            temp = float(val)
        else:
            logger.warning("Parameter %s of file '%s' not recognized!", head, file)
    if temp:
        beta = 1 / temp
    if dt == 0:
        dt = beta / num_timesteps
    elif num_timesteps == 0:
        num_timesteps = int(beta / dt)
    return Parameters(shape, u, eps, t, mu, dt, num_timesteps, warm, meas,
                      num_recomp, prod_len, sampl_recomp)


def log_parameters(p):
    logger.info("_" * 60)
    logger.info("Simulation parameters")
    logger.info("")
    logger.info("     Shape: %s", p.shape)
    logger.info("         U: %s", p.u)
    logger.info("         t: %s", p.t)
    logger.info("       eps: %s", p.eps)
    logger.info("        mu: %s", p.mu)
    logger.info("      beta: %s", p.beta)
    logger.info("      temp: %s", 1 / p.beta)
    logger.info(" time-step: %s", p.dt)
    logger.info("         L: %s", p.num_times)
    logger.info("    nwraps: %s", p.num_wraps)
    logger.info("   prodLen: %s", p.prod_len)
    logger.info("    nequil: %s", p.num_equil)
    logger.info("    nsampl: %s", p.num_sampl)
    logger.info("    recomp: %s", p.sampl_recomp)
    logger.info("      seed: %s", p.seed)
    logger.info("")
