# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import logging
import numpy as np
import matplotlib.pyplot as plt
from dqmc import HubbardModel, BaseDQMC, iteration_fast, mfuncs

logger = logging.getLogger("dqmc")
logger.setLevel(logging.INFO)


class DQMC(BaseDQMC):

    def __init__(self, model, num_timesteps):
        self.gf_up = np.empty([])
        self.gf_dn = np.empty([])
        super().__init__(model, num_timesteps, time_dir=+1)

    def initialize(self):
        self.gf_up, self.gf_dn = self.greens()

    def iteration(self):
        gf_up, gf_dn, accepted = iteration_fast(self.exp_k, self.nu, self.config,
                                                self.bmats_up, self.bmats_dn,
                                                self.gf_up, self.gf_dn, self.times)
        self.gf_up = gf_up
        self.gf_dn = gf_dn
        # Compute and save acceptance ratio
        acc_ratio = accepted / self.config.size
        self.acceptance_probs.append(acc_ratio)

    def get_greens2(self):
        return self.gf_up, self.gf_dn


def main():
    warmup = 1000
    measure = 5000
    model = HubbardModel(10, u=2.0, hop=1.0, beta=2)
    dqmc = DQMC(model, num_timesteps=100)
    occ = dqmc.simulate(warmup, measure, callback=mfuncs.occupation)

    print(occ[0])
    print(occ[1])

    fig, ax = plt.subplots()
    ax.plot(dqmc.acceptance_probs)
    ax.grid()
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlim(0, None)
    ax.set_xlabel("Seeep")
    ax.set_ylabel("Ratio")
    ax.axvline(x=warmup, color="r", ls="--")
    plt.show()


if __name__ == "__main__":
    main()
