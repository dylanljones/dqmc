# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from dqmc import HubbardModel, mfuncs
from dqmc.dqmc import BaseDQMC, iteration_det
from dqmc.time_flow import compute_m_matrices

logger = logging.getLogger("dqmc")
logger.setLevel(logging.INFO)

random.seed(0)


class DQMC(BaseDQMC):

    def __init__(self, model, num_timesteps, time_dir=+1):
        self._old_det = 0
        super().__init__(model, num_timesteps, time_dir)

    def initialize(self):
        m_up, m_dn = compute_m_matrices(self.bmats_up, self.bmats_dn, self.times)
        det_up = np.linalg.det(m_up)
        det_dn = np.linalg.det(m_dn)
        self._old_det = det_up * det_dn

    def iteration(self):
        old_det, accepted = iteration_det(self.exp_k, self.nu, self.config,
                                          self.bmats_up, self.bmats_dn, self._old_det, self.times)
        self._old_det = old_det
        # Compute and save acceptance ratio
        acc_ratio = accepted / self.config.size
        self.acceptance_probs.append(acc_ratio)


def main():
    warmup = 500
    measure = 1000
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
