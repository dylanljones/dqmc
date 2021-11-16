# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import logging
import time as _time
import numpy as np
import scipy.linalg as la
from .config import Configuration


logger = logging.getLogger(__file__)


class Dqmc:

    def __init__(self, model, beta, num_times, warmup=300, sweeps=2000, det_mode=False):
        """ Initialize the Lattice Quantum Monte-Carlo solver.

        Parameters
        ----------
        model : HubbardModel
            The model instance.
        num_times: int
            Number of time steps from .math'0' to .math'\beta'.
        warmup int, optional
            Number of warmup sweeps.
        sweeps: int, optional
            Number of measurement sweeps.
        det_mode: bool, optional
            Flag for the calculation mode. If 'True' the slow algorithm via
            the determinants is used. The default is 'False' (faster).

        """
        self.model = model
        self.config = Configuration.random(model.num_sites, num_times)
        self.warm_sweeps = warmup
        self.meas_sweeps = sweeps

        # Iteration and mode attributes
        self.det_mode = det_mode
        self.status = ""
        self.it = 0
        self.ratio = 0.0
        self.acc = False

        # Cached and temperature-dependend attributes
        self.ham_kin = self.model.hamiltonian_kinetic()
        self.beta = 0.
        self.dtau = 0.
        self.lamb = 0.
        self.exp_k = None
        logger.debug("u=          %s", self.model.u)
        logger.debug("eps=        %s", self.model.eps)
        logger.debug("t=          %s", self.model.hop)
        logger.debug("mu=         %s", self.model.mu)
        logger.debug("num_sites=  %s", self.config.num_sites)
        logger.debug("num_times=  %s", self.config.num_timesteps)
        logger.debug("det_mode=   %s", self.det_mode)
        logger.debug("Warmup=     %s", self.warm_sweeps)
        logger.debug("Measurement=%s", self.meas_sweeps)
        logger.debug("END INIT")

        self.set_beta(beta)

    @property
    def num_sites(self):
        return self.config.num_sites

    @property
    def num_timesteps(self):
        return self.config.num_timesteps

    def set_beta(self, beta: float) -> None:
        """Sets the inverse temperature .math:'\beta' and initializes the calculation.

        Parameters
        ----------
        beta : float
            The inverse of the temperature.
        """
        logger.debug("SETUP")
        self.dtau = beta / self.config.num_timesteps
        self.beta = beta

        self.lamb = np.arccosh(np.exp(self.model.u * self.dtau / 2.)) if self.model.u else 0
        self.exp_k = la.expm(-1 * self.dtau * self.ham_kin)

        logger.debug(f"beta=       %s", self.beta)
        logger.debug(f"dtau=       %s", self.dtau)
        logger.debug(f"lambda=     %s", self.lamb)
        check_val = self.model.u * self.model.hop * self.dtau**2
        if check_val < 0.1:
            logger.info("Check-value %.2f is smaller than 0.1!", check_val)
        else:
            logger.warning("Check-value %.2f should be smaller than 0.1!", check_val)
        logger.debug("END SETUP")

    def set_temperature(self, temp):
        """Sets the temperature and initializes the calculation.

        This is an alternative method to `set_beta`.

        Parameters
        ----------
        temp : float
            The temperature.
        """
        self.set_beta(1 / temp)
    # ===========================================================================================

    def get_exp_v(self, time, sigma):
        r"""Computes the Matrix exponential of .math:'V_\sigma(l)'

        Notes
        -----
        Since .math:'V_\sigma(l) = diag(h_{l, 1}, \dots, h_{l, N}' is a diagonal matrix,
        the numerical matrix exponential is not needed. The exponential of the diagonal elements
        can be computed directly.

        Parameters
        ----------
        time : int
            The time step index.
        sigma : int
            The spin value.

        Returns
        -------
        exp_v : (N, N) np.ndarray
            The matrix exponential of .math:'V_\sigma(l)'.
        """
        diag = -1 * sigma * self.lamb * self.config[:, time]
        return np.diagflat(np.exp(diag))

    def get_m(self, time_0, sigma):
        r"""Computes the 'M' matrices for spin .math:'\sigma'.

        Notes
        -----
        In the warmup loop of the determinant-mode the cyclic permutation is not needed.
        The timeslice 'l0' can be left at 0 to skip the first loop in the computation
        of the matrix-product.

        Parameters
        ----------
        time_0 : int
            Time-slice index used for cyclic permutation.
        sigma : int
            Spin value.

        Returns
        -------
        m: (N, N) np.ndarray
        """
        # Initialize time slices in cyclic permutation:
        # l0, l0-1, ..., 0, L-1, L-2, ..., l0+1
        time_0 = time_0 % self.num_timesteps
        indices = list(reversed(range(self.num_timesteps)))
        time_indices = indices[-time_0:] + indices[:-time_0]

        # compute A=prod(B_l)
        b_prod = 1
        for time in time_indices:
            exp_v = self.get_exp_v(time, sigma)
            b = np.dot(self.exp_k, exp_v)
            b_prod = np.dot(b_prod, b)

        # Assemble M=I+prod(B)
        return np.eye(self.num_sites) + b_prod

    def iter_sweeps(self, n, console_updates=200):
        """Iterates over the specified number of sweeps.

        The generator is mainly used for logging and updating the console.

        Parameters
        ----------
        n : int
            Number of sweeps.
        console_updates : int, optional
            Number of times the console output is updated during all sweeps. The default is '200'.

        Yields
        ------
        it: int
            Current iteration index.
        """
        logger.debug(self.status.upper())
        print_interval = max(1, int(n/console_updates))
        for it in range(n):
            count = it + 1
            if it < n and count % print_interval == 0:
                string = f"{self.status} Sweep {it + 1} ({100 * count / n:.1f}%)"
                string += f" [Mean: {self.config.mean():5.2f}, Var: {self.config.var():5.2f}]"
                print("\r" + string, end="", flush=True)
            self.it = it
            yield it
        print(f"\r{self.status} Sweep {n} (100.0%)")
        logger.debug("END " + self.status.upper())

    def _log_iter(self, site, time):
        """ Logs the attributes of the iteration.

        The format of logs is:
        <Status> <iteration> -- <time_slice> <site> -- <ratio> (<accepted>) -- <mc mean> <mc var>

        Parameters
        ----------
        site : int
            Site index.
        time : int
            Time-slice index.
        """
        log = f"{self.status} {self.it + 1} -- {time:>2} {site:>2} -- {self.ratio:.1f} ({self.acc})"
        log += f" -- {self.config.mean():.3f} {self.config.var():.3f}"
        logger.debug(log)

    # =========================================================================

    def _update_step_det(self, old_det):
        # Iterate over all time-steps, starting at the end (.math:'\beta')
        for time in reversed(range(self.num_timesteps)):
            # Iterate over all lattice sites
            for site in range(self.num_sites):
                # Update Configuration by flipping spin
                self.config.update(site, time)
                # Compute updated M matrices after config-update
                m_up = self.get_m(time, sigma=+1)
                m_dn = self.get_m(time, sigma=-1)
                # Compute the new determinant for both matrices for the acceptance ratio
                new_det = np.linalg.det(m_up) * np.linalg.det(m_dn)
                self.ratio = new_det / old_det
                self.acc = np.random.rand() <= self.ratio
                if self.acc:
                    # Move accepted:
                    # Continue using the new configuration
                    old_det = new_det
                else:
                    # Move not accepted:
                    # Revert to the old configuration by updating again
                    self.config.update(site, time)
                self._log_iter(site, time)
        return old_det

    def warmup_loop_det(self):
        """ Runs the slow version of the LQMC warmup-loop """
        self.status = "Warmup"
        # Calculate M matrices for both spins
        m_up = self.get_m(0, sigma=+1)
        m_dn = self.get_m(0, sigma=-1)
        # Initialize the determinant for both matrices
        old_det = np.linalg.det(m_up) * np.linalg.det(m_dn)
        # Warmup-sweeps
        for _ in self.iter_sweeps(self.warm_sweeps):
            old_det = self._update_step_det(old_det)

    def measure_loop_det(self):
        r""" Runs the slow version of the LQMC measurement-loop and returns the Green's function.

        Returns
        -------
        gf : (2, N, N) np.ndarray
            Measured Green's function .math:'G' of the up- and down-spin channel.
        """
        self.status = "Measurement"
        # Calculate M matrices for both spins
        m_up = self.get_m(0, sigma=+1)
        m_dn = self.get_m(0, sigma=-1)
        # Initialize the determinant for both matrices
        old_det = np.linalg.det(m_up) * np.linalg.det(m_dn)
        # Initialize greens functions
        shape = (self.num_sites, self.num_sites)
        gf_total_up = np.zeros(shape, dtype=np.float64)
        gf_total_dn = np.zeros(shape, dtype=np.float64)
        # Measurement-sweeps
        for _ in self.iter_sweeps(self.meas_sweeps):
            old_det = self._update_step_det(old_det)
            # Perform measurements
            m_up = self.get_m(0, sigma=+1)
            m_dn = self.get_m(0, sigma=-1)
            gf_total_up += np.linalg.inv(m_up)
            gf_total_dn += np.linalg.inv(m_dn)
        # Return the normalized total green functions
        return np.asarray([gf_total_up, gf_total_dn]) / self.meas_sweeps

    # =========================================================================

    def _update_step(self):
        # Compute M matrices
        m_up = self.get_m(0, sigma=+1)
        m_dn = self.get_m(0, sigma=-1)
        # Initialize greens functions
        gf_up = np.linalg.inv(m_up)
        gf_dn = np.linalg.inv(m_dn)
        # Iterate over all time-steps, starting at the end (.math:'\beta')
        for time in reversed(range(self.num_timesteps)):
            # Iterate over all lattice sites
            for i in range(self.num_sites):
                # Compute acceptance ratio
                arg = 2 * self.lamb * self.config[i, time]
                d_up = 1 + (1 - gf_up[i, i]) * (np.exp(+arg) - 1)
                d_dn = 1 + (1 - gf_dn[i, i]) * (np.exp(-arg) - 1)
                self.ratio = d_up * d_dn
                self.acc = np.random.rand() <= self.ratio
                if self.acc:
                    # Update Greens function
                    c_up = -(np.exp(-arg) - 1) * gf_up[i, :]
                    c_up[i] += (np.exp(-arg) - 1)
                    c_dn = -(np.exp(+arg) - 1) * gf_dn[i, :]
                    c_dn[i] += (np.exp(+arg) - 1)
                    # These are the bs in the tutorial, but I call them e
                    # here to avoid confusing them with the B-matrices
                    e_up = gf_up[:, i] / (1 + c_up[i])
                    e_dn = gf_dn[:, i] / (1 + c_dn[i])
                    for j in range(self.num_sites):
                        for k in range(self.num_sites):
                            gf_up[j, k] = gf_up[j, k] - e_up[j] * c_up[k]
                            gf_dn[j, k] = gf_dn[j, k] - e_dn[j] * c_dn[k]
                    # Update HS-field
                    self.config.update(i, time)

                self._log_iter(i, time)

            # Update the GF for the next time slice (Wrapping)
            if time > 0:  # Only do this, if this is not the last l-loop
                exp_v_up = self.get_exp_v(time - 1, sigma=+1)
                exp_v_dn = self.get_exp_v(time - 1, sigma=-1)
                b_up = np.dot(exp_v_up, self.exp_k)
                b_dn = np.dot(exp_v_dn, self.exp_k)

                gf_up = np.dot(np.dot(b_up, gf_up), np.linalg.inv(b_up))
                gf_dn = np.dot(np.dot(b_dn, gf_dn), np.linalg.inv(b_dn))

        return gf_up, gf_dn

    def warmup_loop(self):
        """ Runs the fast version of the LQMC warmup-loop """
        self.status = "Warmup"
        # Warmup-sweeps
        for _ in self.iter_sweeps(self.warm_sweeps):
            self._update_step()

    def measure_loop(self):
        r""" Runs the fast version of the LQMC measurement-loop and returns the Green's function.
        gf: (2, N, N) np.ndarray
            Measured Green's function .math'G' of the up- and down-spin channel.
        """
        self.status = "Measurement"
        # Initialize greens functions
        shape = (self.num_sites, self.num_sites)
        gf_total_up = np.zeros(shape, dtype=np.float64)
        gf_total_dn = np.zeros(shape, dtype=np.float64)
        # Measurement-sweeps
        for _ in self.iter_sweeps(self.meas_sweeps):
            # Initialize greens functions
            gf_up, gf_dn = self._update_step()
            # Perform measurements
            gf_total_up += gf_up
            gf_total_dn += gf_dn
        # Return the normalized total green functions
        return np.asarray([gf_total_up, gf_total_dn]) / self.meas_sweeps

    # =========================================================================

    def run_lqmc(self):
        """Runs the warmup and measurment loop and returns the Green's function.

        Returns
        -------
        gf : (2, N, N) np.ndarray
            Measured Green's function .math'G' of the up- and down-spin channel.
        """
        t0 = _time.time()
        if self.det_mode:
            self.warmup_loop_det()
            gf = self.measure_loop_det()
        else:
            self.warmup_loop()
            gf = self.measure_loop()
        mins, secs = divmod(_time.time() - t0, 60)
        logger.info(f"Total time: {int(mins):0>2}:{int(secs):0>2} min")
        return gf
