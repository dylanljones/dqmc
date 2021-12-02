# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones

import logging
import numpy as np
import matplotlib.pyplot as plt
from dqmc import HubbardModel, BaseDQMC, iteration_fast, mfuncs, compute_m_matrices, iteration_det
from tqdm import tqdm

logger = logging.getLogger("dqmc")
logger.setLevel(logging.WARNING)


class DetDQMC(BaseDQMC):

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


class FastDQMC(BaseDQMC):

    def __init__(self, model, num_timesteps):
        self.gf_up = np.empty([])
        self.gf_dn = np.empty([])
        super().__init__(model, num_timesteps, time_dir=+1)

    def initialize(self):
        self.gf_up, self.gf_dn = self.greens()

    def iteration(self):
        accepted = iteration_fast(self.exp_k, self.nu, self.config, self.bmats_up, self.bmats_dn,
                                  self.gf_up, self.gf_dn, self.times)
        # Compute and save acceptance ratio
        acc_ratio = accepted / self.config.size
        self.acceptance_probs.append(acc_ratio)

    def get_greens(self):
        return self.gf_up, self.gf_dn


class DQMC(BaseDQMC):

    def __init__(self, model, num_timesteps, mode="fast"):
        self._gf_up = np.empty([])
        self._gf_dn = np.empty([])
        self._old_det = 0.
        self.mode = mode
        self._iter = self.iteration_fast if mode == "fast" else self.iteration_det
        super().__init__(model, num_timesteps, time_dir=+1)

    def initialize(self):
        m_up, m_dn = compute_m_matrices(self.bmats_up, self.bmats_dn, self.times)
        det_up = np.linalg.det(m_up)
        det_dn = np.linalg.det(m_dn)
        self._old_det = det_up * det_dn
        self._gf_up = np.linalg.inv(m_up)
        self._gf_dn = np.linalg.inv(m_dn)

    def iteration_fast(self):
        accepted = iteration_fast(self.exp_k, self.nu, self.config, self.bmats_up, self.bmats_dn,
                                  self._gf_up, self._gf_dn, self.times)

        return accepted

    def iteration_det(self):
        old_det, accepted = iteration_det(self.exp_k, self.nu, self.config,
                                          self.bmats_up, self.bmats_dn, self._old_det, self.times)
        self._old_det = old_det
        return accepted

    def iteration(self):
        accepted = self._iter()
        # Compute and save acceptance ratio
        acc_ratio = accepted / self.config.size
        self.acceptance_probs.append(acc_ratio)

    def get_greens(self):
        if self.mode == "fast":
            return self._gf_up, self._gf_dn
        else:
            return self.greens()


def plot_acceptance_probs(probs, warmup):
    fig, ax = plt.subplots()
    ax.plot(probs)
    ax.grid()
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlim(0, None)
    ax.set_xlabel("Seeep")
    ax.set_ylabel("Ratio")
    ax.axvline(x=warmup, color="r", ls="--")


def main():
    u, mu, hop = 4.0, 0.0, 1.0
    num_sites = 10
    num_timesteps = 50
    warmup = 500
    measure = 3000

    temps = np.geomspace(0.2, 100, 50)
    moments = np.zeros((len(temps), 10))
    for i in tqdm(range(len(temps))):
        model = HubbardModel(num_sites, u=u, mu=mu, hop=hop, beta=1 / temps[i])
        dqmc = FastDQMC(model, num_timesteps)
        mom = dqmc.simulate(warmup, measure, callback=mfuncs.mz_moment)
        moments[i] = mom

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.plot(temps, moments.T[4])
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\langle m_z^2 \rangle$")
    plt.show()


if __name__ == "__main__":
    main()
