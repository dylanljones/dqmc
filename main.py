# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dqmc import parse, run_dqmc, map_params, run_dqmc_parallel

logger = logging.getLogger("dqmc")
# logger.setLevel(logging.INFO)


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
    logger.info("         L: %s", p.num_timesteps)
    logger.info("   nrecomp: %s", p.num_recomp)
    logger.info("   prodLen: %s", p.prod_len)
    logger.info("    nequil: %s", p.num_equil)
    logger.info("    nsampl: %s", p.num_sampl)
    logger.info("      seed: %s", p.seed)
    logger.info("")


def log_results(*results):
    n_up = np.mean(results[0])
    n_dn = np.mean(results[1])
    n_double = np.mean(results[2])
    local_moment = np.mean(results[3])

    logger.info("_" * 60)
    logger.info("Simulation results")
    logger.info("")
    logger.info("     Total density: %8.4f", n_up + n_dn)
    logger.info("   Spin-up density: %8.4f", n_up)
    logger.info(" Spin-down density: %8.4f", n_dn)
    logger.info("  Double occupancy: %8.4f", n_double)
    logger.info("      Local moment: %8.4f", local_moment)
    if results[4]:
        logger.info("   Callback results: %s", results[4])
    logger.info("")


def main():
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
        x = list(kwargs.values())[0]
        y = [np.mean(res[i]) for res in results]
        plt.plot(x, y)
        plt.show()
    else:
        logger.setLevel(logging.INFO)

        log_parameters(p)
        logger.info("Starting DQMC simulation...")
        results = run_dqmc(p)
        log_results(*results)


if __name__ == "__main__":
    main()
