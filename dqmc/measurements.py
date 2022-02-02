# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass
from .dqmc import compute_unequal_time_greens

gmat_t = float64[:, ::1]
gtau_t = float64[:, :, ::1]


@jitclass([
    ("count", int64),
    ("num_sites", int64),
    ("_gf_up", gmat_t),
    ("_gf_dn", gmat_t),
    ("_n_up", float64[:]),
    ("_n_dn", float64[:]),
    ("_n_dbl", float64[:]),
    ("_mz2", float64[:]),
    ("_gftau0_up", gtau_t),
    ("_gftau0_dn", gtau_t),
])
class MeasurementData:
    """Measurement container for equal time observables."""
    def __init__(self, num_sites, num_times):
        self.count = 0
        self.num_sites = num_sites

        # Initialize measurement data
        self._gf_up = np.zeros((num_sites, num_sites), dtype=np.float64)
        self._gf_dn = np.zeros((num_sites, num_sites), dtype=np.float64)
        self._n_up = np.zeros(num_sites, dtype=np.float64)
        self._n_dn = np.zeros(num_sites, dtype=np.float64)
        self._n_dbl = np.zeros(num_sites, dtype=np.float64)
        self._mz2 = np.zeros(num_sites, dtype=np.float64)

        self._gftau0_up = np.zeros((num_times, num_sites, num_sites), dtype=np.float64)
        self._gftau0_dn = np.zeros((num_times, num_sites, num_sites), dtype=np.float64)

    @property
    def gf_up(self):
        return self._gf_up / self.count

    @property
    def gf_dn(self):
        return self._gf_dn / self.count

    @property
    def n_up(self):
        return self._n_up / self.count

    @property
    def n_dn(self):
        return self._n_dn / self.count

    @property
    def n_dbl(self):
        return self._n_dbl / self.count

    @property
    def mz2(self):
        return self._mz2 / self.count

    @property
    def gftau0_up(self):
        return self._gftau0_up / self.count

    @property
    def gftau0_dn(self):
        return self._gftau0_dn / self.count

    def accumulate(self, gf_up, gf_dn, sgns):
        """Acummulate (unnormalized) equal time measurements of observables.

        Parameters
        ----------
        gf_up : (N, N) np.ndarray
            The current spin-up Green's function matrix :math:`G_↑`.
        gf_dn : (N, N) np.ndarray
            The current spin-down Green's function matrix :math:`G_↓`.
        sgns : (2,) np.ndarray
            The sign of the determinant of the Green's function of both spins.

        Notes
        -----
        The measured observables are defined as
        ..math::
            <n_{iσ}>  = 1 - [G_σ]_{ii}
            <n_↑ n_↓> = (1 - [G_↑]_{ii}) (1 - [G_↓]_{ii})
            <m_z^2> = <n_↑> + <n_↓> - 2 <n_↑ n_↓>
        """
        signfac = sgns[0] * sgns[1]

        n_up = 1 - np.diag(gf_up)
        n_dn = 1 - np.diag(gf_dn)
        n_dbl = n_up * n_dn
        mz2 = n_up + n_dn - 2 * n_dbl

        self.count += 1
        self._gf_up += gf_up * signfac
        self._gf_dn += gf_dn * signfac
        self._n_up += n_up * signfac
        self._n_dn += n_dn * signfac
        self._n_dbl += n_dbl * signfac
        self._mz2 += mz2 * signfac

    def accumulate_unequal_time(self, bmats_up, bmats_dn, gf_up, gf_dn, sgns):
        signfac = sgns[0] * sgns[1]
        self._gftau0_up += signfac * compute_unequal_time_greens(bmats_up, gf_up)
        self._gftau0_dn += signfac * compute_unequal_time_greens(bmats_dn, gf_dn)

    def normalize(self):
        """Normalizes and returns all equal time measurements of observables."""
        gf_up = self._gf_up / self.count
        gf_dn = self._gf_dn / self.count
        n_up = self._n_up / self.count
        n_dn = self._n_dn / self.count
        n_dbl = self._n_dbl / self.count
        mz2 = self._mz2 / self.count
        gftau0_up = self._gftau0_up / self.count
        gftau0_dn = self._gftau0_dn / self.count
        return gf_up, gf_dn, n_up, n_dn, n_dbl, mz2, gftau0_up, gftau0_dn
