# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import sys
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
    logger.info("      temp: %s", 1 / p.beta)
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


def main():
    args = sys.argv[1:]
    if len(args):
        file = args[0]
    else:
        file = "test.txt"

    p = parse(file)
    model = hubbard_hypercube(p.shape, p.u, p.eps, p.t, p.mu, p.beta)

    log_parameters(p)

    logger.info("Starting DQMC simulation...")
    sim = DQMC(model, p.num_timesteps, p.num_recomp, p.prod_len, seed=p.seed)
    sim.simulate(p.num_equil, p.num_sampl)

    log_results(sim)


if __name__ == "__main__":
    main()
