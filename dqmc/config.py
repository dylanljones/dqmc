# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from typing import Union, Optional, Tuple


class Configuration:
    """ Configuration class representing the discrete Hubbard-Stratonovich (HS) field."""

    def __init__(self, num_sites: Optional[int] = 0, num_timesteps: Optional[int] = 1,
                 array: Optional[np.ndarray] = None, dtype: Optional[Union[str, np.dtype]] = np.int8):
        """Initializes the discrete Hubbard-Stratonovich (HS) field."""
        if array is None:
            # Create an array of random 0 and 1 and scale array to -1 and 1
            num_timesteps = max(1, num_timesteps)
            array = 2 * np.random.randint(0, 2, size=(num_sites, num_timesteps)) - 1
        self._config = np.asarray(array, dtype=dtype)

    @property
    def shape(self) -> Tuple[int, int]:
        return self._config.shape

    @property
    def dtype(self) -> np.dtype:
        return self._config.dtype

    @property
    def num_sites(self) -> int:
        return self._config.shape[0]

    @property
    def num_timesteps(self) -> int:
        return self._config.shape[1]

    @property
    def config(self) -> np.ndarray:
        return self._config

    def shuffle(self) -> None:
        """Fills the configuration with a random distribution of (-1, +1)"""
        self._config[:] = 2 * np.random.randint(0, 2, size=self._config.shape) - 1

    def update(self, i: int, t: Optional[int] = 0) -> None:
        """ Updates an element of the HS-field by flipping its spin-value.

        Parameters
        ----------
        i: int
            The index of the site.
        t: int, optional
            The index of the time slice.
        """
        self._config[i, t] *= -1

    def mean(self) -> float:
        """Computes the Monte-Carlo sample mean."""
        return float(np.mean(self._config))

    def var(self) -> float:
        """Computes the Monte-Carlo sample variance."""
        return float(np.var(self._config))

    def reshape(self, shape: Union[int, Tuple[int]], order: Optional[str] = "C") -> np.ndarray:
        """Reshapes the HS-field and returns the result as a ``np.ndarray``.

        Parameters
        ----------
        shape: int or tuple of int
            The shape of the output array.
        order: str, optional
            Optional ordering for the reshape. The default is ``C``.

        Returns
        -------
        array: np.ndarray
        """
        return np.reshape(self.config, shape, order)

    def pformat(self, delim: Optional[str] = " ") -> str:
        """Returns a formated string of the configuration."""
        header = r"i\l  " + delim.join([f"{i:^3}" for i in range(self.num_timesteps)])
        rows = list()
        for site in range(self.num_sites):
            row = delim.join([f"{x:^3}" for x in self._config[site, :]])
            rows.append(f"{site:<3} [{row}]")
        return header + "\n" + "\n".join(rows)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape: {self.shape}, dtype={self.dtype})"

    def __str__(self) -> str:
        return self.pformat()
