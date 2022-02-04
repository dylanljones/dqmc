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

# maps label to attribute name and types
ATTR_LABEL_MAP = {
    "shape": [("shape", ), int, 5],
    "u": [("u", ), float, 0.0],
    "eps": [("eps", ), float, 0.0],
    "t": [("t", "hop"), float, 1.0],
    "mu": [("mu", ), float, 0.0],
    "dt": [("dt", ), float, 0.1],
    "beta": [("beta",), float],
    "temp": [("temp",), float],
    "num_times": [("l", "num_times"), int, 40],
    "num_equil": [("nequil", "num_equil"), int, 512],
    "num_sampl": [("nsampl", "num_sampl"), int, 512],
    "num_wraps": [("nwraps", "num_wraps"), int, 0],
    "prod_len": [("prodlen", "prod_len"), int, 0],
    "sampl_recomp": [("recomp", "sampl_recomp"), int, 0],
    "seed": [("seed", ), int, 0],
}


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


def _build_attribute_map():
    attr_map = dict()
    for attr, info in ATTR_LABEL_MAP.items():
        keys = info[0]
        attr_type = info[1]
        default = None if len(info) == 2 else info[2]
        for key in keys:
            if key in attr_map:
                raise ValueError(f"Key {key} already registered in attribute map!")
            attr_map[key] = [attr, attr_type, default]
    return attr_map


def _read_param_file(file):
    # Initialize attribute map
    attr_map = _build_attribute_map()
    # Fill items with default values
    items = dict()
    for (attr, _, default) in attr_map.values():
        if default is not None:
            items[attr] = default
    # Read file content
    with open(file, "r") as fh:
        text = fh.read()
    # Parse lines of file
    for line in text.splitlines(keepends=False):
        # Extract label and value
        if "#" in line:
            text, comm = line.strip().split("#")
            line = text.strip()
        if not line:
            continue
        label, data = line.split(maxsplit=1)
        label = label.lower()
        data = data.replace(",", "").split(" ")
        try:
            # Parse value and cast to type
            template = attr_map[label]
            key = template[0]
            datatype = template[1]
            values = [(datatype(data[i])) for i in range(len(data))]
            # Store values in dictionary
            items[key] = values if len(values) > 1 else values[0]
        except KeyError:
            print("Parameter %s of file '%s' not recognized!" % (label, file))

    return items


def parse(file):
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
    items = _read_param_file(file)
    if "temp" in items:
        items["beta"] = 1 / items["temp"]
    if items.get("dt") == 0:
        items["dt"] = items["beta"] / items["num_times"]
    elif items["num_times"] == 0:
        items["num_times"] = int(items["beta"] / items["dt"])
    items.pop("beta")
    items.pop("temp")
    return Parameters(**items)


def parse2(file):  # noqa: C901
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
