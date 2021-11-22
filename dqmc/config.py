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
from typing import Union, Tuple

UP, DN = +1, -1


class Configuration:

    """Configuration class representing the Hubbard-Stratonovich (HS) field."""

    def __init__(self, inputarr, dtype=None):
        self._config = np.asarray(inputarr, dtype=dtype)
        if len(self.shape) != 2:
            raise ValueError(f"Inputarray must be 2 dimensional, not {len(self.shape)}D!")

    @classmethod
    def random(cls, num_sites: int = 0, num_timesteps: int = 1,
               dtype: Union[str, np.dtype] = np.int8):
        """Initializes the HS-field with a random distribution of (-1, +1)."""
        num_timesteps = max(1, num_timesteps)
        array = 2 * np.random.randint(0, 2, size=(num_sites, num_timesteps)) - 1
        return cls(array, dtype)

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns the shape of the HS field."""
        return self._config.shape

    @property
    def size(self) -> int:
        """Returns the toal number of elements in the configuration array."""
        return self._config.shape[0] * self._config.shape[1]

    @property
    def num_sites(self) -> int:
        """The number of sites used in the configuration."""
        return self._config.shape[0]

    @property
    def num_timesteps(self) -> int:
        """The number of time steps used in the configuration."""
        return self._config.shape[1]

    def shuffle(self) -> None:
        """Fills the configuration with a random distribution of (-1, +1)."""
        self._config[:] = 2 * np.random.randint(0, 2, size=self.shape) - 1

    def update(self, site: int, time: int = 0) -> None:
        """Updates an element of the HS-field by flipping its spin-value.

        Parameters
        ----------
        site : int
            The index of the site.
        time : int, optional
            The index of the time slice. The default is `0`.
        """
        self._config[site, time] *= -1

    def mean(self, *args, **kwargs):
        """Computes the mean of the HS field."""
        return self._config.mean(*args, **kwargs)

    def var(self, *args, **kwargs):
        """Computes the variance of the HS field."""
        return self._config.var(*args, **kwargs)

    def pformat(self, delim: str = " ") -> str:
        """Returns a formated string of the configuration."""
        header = r"i\l  " + delim.join([f"{i:^3}" for i in range(self.num_timesteps)])
        rows = list()
        for site in range(self.num_sites):
            row = delim.join([f"{x:^3}" for x in self._config[site, :]])
            rows.append(f"{site:<3} [{row}]")
        return header + "\n" + "\n".join(rows)

    def copy(self):
        """Creates a deep copy of the configuration array."""
        return Configuration(self._config.copy())

    def __getitem__(self, item):
        return self._config[item]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape: {self.shape}, dtype={self._config.dtype})"

    def __str__(self) -> str:
        return self.pformat()
