# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import sys
import time
import logging
from dqmc import hubbard_hypercube
from dqmc.simulator import DQMC
from dqmc.parser import parse


logger = logging.getLogger("dqmc")
logger.setLevel(logging.INFO)


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


def pformat_results(n_up, n_dn, n_double, local_moment):
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
        file = "sample_chain.txt"

    p = parse(file)
    model = hubbard_hypercube(p.shape, p.u, p.eps, p.t, p.mu, p.beta)

    seed = 0  # random.randint(0, 100_000_000)
    print("Random seed:", seed)
    print(pformat_parameters(p))

    t0 = time.perf_counter()
    print("Starting DQMC simulation...")
    sim = DQMC(model, p.num_timesteps, num_recomp=p.num_recomp, seed=seed)
    sim.simulate(p.num_equil, p.num_sampl)

    print(f"{p.num_equil + p.num_sampl} iterations completed!")
    print(f"CPU time: {time.perf_counter() - t0:.2f}s")
    print(pformat_results(sim.n_up, sim.n_dn, sim.n_double, sim.local_moment))


if __name__ == "__main__":
    main()
