# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import logging
from typing import Union
from dataclasses import dataclass

logger = logging.getLogger("dqmc")
logger.setLevel(logging.WARNING)


@dataclass
class Parameters:

    shape: Union[int, tuple]
    u: float
    eps: float
    t: float
    mu: float
    dt: float
    num_timesteps: int
    num_equil: int
    num_sampl: int
    num_recomp: int

    @property
    def beta(self):
        return self.num_timesteps * self.dt


def parse(file):
    shape = 0
    u = 0.
    eps = 0.
    t = 0.
    mu = 0.
    dt = 0.
    num_timesteps = 0
    warm = 0
    meas = 0
    num_recomp = 0

    with open(file, "r") as fh:
        text = fh.read()
    lines = text.splitlines(keepends=False)
    for line in lines:
        if line.strip().startswith("#"):
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
        elif head == "nrecomp":
            num_recomp = int(val)
        else:
            logger.warning("Parameter %s of file '%s' not recognized!", head, file)

    return Parameters(shape, u, eps, t, mu, dt, num_timesteps, warm, meas, num_recomp)
