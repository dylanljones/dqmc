# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dqmc import (
    parse,
    run_dqmc,
    map_params,
    run_dqmc_parallel,
    log_parameters,
    log_results
)

logger = logging.getLogger("dqmc")


# noinspection PyShadowingBuiltins
def parse_array_args(strings, type=float):
    if "..." in strings:
        a, b = type(strings[0]), type(strings[-1])
        if len(strings) == 3:
            step = 1.0
        else:
            step = type(strings[1]) - a
        values = np.arange(a, b + 0.1 * step, step)
    else:
        values = [type(s) for s in strings]
    return np.array(values, dtype=type)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--processes", "-mp", type=int, default=1)
    parser.add_argument("-hf", action="store_true")
    parser.add_argument("-u", type=str, nargs="+")
    parser.add_argument("-eps", type=str, nargs="+")
    parser.add_argument("-t", type=str, nargs="+")
    parser.add_argument("-mu", type=str, nargs="+")
    parser.add_argument("-dt", type=str, nargs="+")
    parser.add_argument("-temp", type=str, nargs="+")
    parser.add_argument("--plot", "-p", type=str, default="moment",
                        choices=["nup", "ndn", "n2", "moment"])
    args = parser.parse_args(argv)
    argdict = dict(args.__dict__)

    p = parse(argdict.pop("file"))
    hf = argdict.pop("hf")
    kwargs = dict()
    plot = argdict.pop("plot")
    processes = argdict.pop("processes")
    if processes == -1:
        processes = None
    # Parse arguments
    for key in argdict.keys():
        if argdict[key] is not None:
            kwargs[key] = parse_array_args(argdict[key], type=float)

    # Half filling
    if hf and "mu" not in kwargs:
        if "u" in kwargs and "eps" not in kwargs:
            kwargs["mu"] = np.array(kwargs["u"]) / 2 - p.eps
        elif "eps" in kwargs and "u" not in kwargs:
            kwargs["mu"] = p.u / 2 - np.array(kwargs["eps"])
        elif "u" in kwargs and "eps" in kwargs:
            kwargs["mu"] = np.array(kwargs["u"]) / 2 - np.array(kwargs["eps"])

    return p, kwargs, plot, processes


args = sys.argv[1:]
if len(args) == 0:
    argstr = "test.txt"
    args = argstr.split(" ")
p, kwargs, plot, max_workers = parse_args(args)

if kwargs:
    logger.setLevel(logging.WARNING)

    params = map_params(p, **kwargs)
    results = run_dqmc_parallel(params, max_workers=max_workers)
    i = ["nup", "ndn", "n2", "moment"].index(plot)
    xlabel = list(kwargs.keys())[0]
    x = kwargs[xlabel]
    y = [np.mean(res[i]) for res in results]
    ylabel = plot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(x, y)
    plt.show()
else:
    logger.setLevel(logging.DEBUG)

    log_parameters(p)
    logger.info("Starting DQMC simulation...")
    results = run_dqmc(p)
    log_results(*results)
