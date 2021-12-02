# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones


import logging
import numpy as np
import matplotlib.pyplot as plt
from dqmc import HubbardModel, BaseDQMC, iteration_fast, mfuncs, compute_m_matrices, iteration_det
from dqmc import compute_greens

logger = logging.getLogger("dqmc")
logger.setLevel(logging.INFO)


def plot_acceptance_probs(probs, warmup):
    fig, ax = plt.subplots()
    ax.plot(probs)
    ax.grid()
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlim(0, None)
    ax.set_xlabel("Seeep")
    ax.set_ylabel("Ratio")
    ax.axvline(x=warmup, color="r", ls="--")


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
        self._gf_up = np.ascontiguousarray(np.linalg.inv(m_up))
        self._gf_dn = np.ascontiguousarray(np.linalg.inv(m_dn))

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
        if False and self.mode == "fast":
            return self._gf_up, self._gf_dn
        else:
            return self.greens()


def run_dqmc(num_sites, num_timesteps, u, temp, mu=0., hop=1., warmup=500, measure=2000,
             callback=None):
    model = HubbardModel(num_sites, u=u, mu=mu, hop=hop, beta=1 / temp)
    dqmc = DQMC(model, num_timesteps, mode="fast")
    return dqmc.simulate(warmup, measure, callback)


def pformat_result(value, error, dec=5):
    return f"{value:.{dec}f}±{error:.{dec}f}"


def main():
    num_sites = 10
    num_timesteps = 50
    warmup = 500
    measure = 3000
    u, mu, hop = 4.0, 0.0, 1.0
    temp = 1

    res = run_dqmc(num_sites, num_timesteps, u, temp, mu, hop, warmup, measure, mfuncs.mz_moment)
    print(res)
    print(pformat_result(np.mean(res), np.std(res), dec=4))


if __name__ == "__main__":
    main()

