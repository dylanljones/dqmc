# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
import logging
import random
from dqmc import hubbard_hypercube
from dqmc.simulator import DQMC
from dqmc.parser import parse


logger = logging.getLogger("dqmc")
logger.setLevel(logging.DEBUG)


def pformat_parameters(p):
    pass


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
    p = parse("sample_chain.txt")

    model = hubbard_hypercube(p.shape, p.u, p.eps, p.t, p.mu, p.beta)

    seed = random.randint(0, 100_000_000)
    print("Random seed:", seed)
    print(p)

    sim = DQMC(model, p.num_timesteps, seed=seed)
    sim.simulate(p.num_equil, p.num_sampl)

    print(pformat_results(sim.n_up, sim.n_dn, sim.n_double, sim.local_moment))


if __name__ == "__main__":
    main()
