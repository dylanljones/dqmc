# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import sys
import time
import logging
import numpy as np
from dqmc import hubbard_hypercube
from dqmc.simulator import DQMC
from dqmc.parser import parse


logger = logging.getLogger("dqmc")
logger.setLevel(logging.DEBUG)


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
    logger.info(" time-step: %s", p.dt)
    logger.info("         L: %s", p.num_timesteps)
    logger.info("   nrecomp: %s", p.num_recomp)
    logger.info("   prodLen: %s", p.prod_len)
    logger.info("    nequil: %s", p.num_equil)
    logger.info("    nsampl: %s", p.num_sampl)
    logger.info("      seed: %s", p.seed)
    logger.info("")


def log_results(sim):
    n_up = np.mean(sim.n_up)
    n_dn = np.mean(sim.n_dn)
    n_double = np.mean(sim.n_double)
    local_moment = np.mean(sim.local_moment)

    logger.info("_" * 60)
    logger.info("Simulation results")
    logger.info("")
    logger.info("     Total density: %8.4f", n_up + n_dn)
    logger.info("   Spin-up density: %8.4f", n_up)
    logger.info(" Spin-down density: %8.4f", n_dn)
    logger.info("  Double occupancy: %8.4f", n_double)
    logger.info("      Local moment: %8.4f", local_moment)
    logger.info("")


def pformat_parameters(p):
    line = "_" * 60
    s = line + "\n"
    s += "Simulation parameters\n\n"
    s += f"            Shape: {p.shape}\n"
    s += f"                u: {p.u}\n"
    s += f"              eps: {p.eps}\n"
    s += f"                t: {p.t}\n"
    s += f"               mu: {p.mu}\n"
    s += f"             beta: {p.beta}\n"
    s += f"        time-step: {p.dt}\n"
    s += f"                L: {p.num_timesteps}\n"
    s += f"           nequil: {p.num_equil}\n"
    s += f"           nsampl: {p.num_sampl}\n"
    s += f"          nrecomp: {p.num_recomp}\n"
    return s


def pformat_results(sim):
    n_up = np.mean(sim.n_up)
    n_dn = np.mean(sim.n_dn)
    n_double = np.mean(sim.n_double)
    local_moment = np.mean(sim.local_moment)

    line = "_" * 60
    s = line + "\n"
    s += "Simulation results\n\n"
    s += f"     Total density: {n_up + n_dn:8.4f}\n"
    s += f"   Spin-up density: {n_up:8.4f}\n"
    s += f" Spin-down density: {n_dn:8.4f}\n"
    s += f"  Double occupancy: {n_double:8.4f}\n"
    s += f"      Local moment: {local_moment:8.4f}\n"
    return s


def main():
    args = sys.argv[1:]
    if len(args):
        file = args[0]
    else:
        file = "tmp.txt"

    p = parse(file)
    model = hubbard_hypercube(p.shape, p.u, p.eps, p.t, p.mu, p.beta)

    log_parameters(p)
    # print(pformat_parameters(p))

    logger.info("Starting DQMC simulation...")
    t0 = time.perf_counter()
    sim = DQMC(model, p.num_timesteps, p.num_recomp, p.prod_len, seed=p.seed)
    sim.simulate(p.num_equil, p.num_sampl)
    t = time.perf_counter() - t0

    logger.info("%s iterations completed!", p.num_equil + p.num_sampl)
    logger.info("CPU time: %.2fs", t)
    log_results(sim)
    # print(pformat_results(sim.n_up, sim.n_dn, sim.n_double, sim.local_moment))


if __name__ == "__main__":
    main()
