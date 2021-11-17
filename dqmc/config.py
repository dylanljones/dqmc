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
from typing import Union

UP, DN = +1, -1


class Configuration:

    def __init__(self, inputarr, dtype=None):
        self._config = np.asarray(inputarr, dtype=dtype)
        if len(self.shape) != 2:
            raise ValueError(f"Inputarray must be 2 dimensional, not {len(self.shape)}D!")

    @classmethod
    def random(cls, num_sites: int = 0, num_timesteps: int = 1,
               dtype: Union[str, np.dtype] = np.int8):
        num_timesteps = max(1, num_timesteps)
        array = 2 * np.random.randint(0, 2, size=(num_sites, num_timesteps)) - 1
        return cls(array, dtype)

    @property
    def shape(self):
        return self._config.shape

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

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *args, **kwargs):
        return self._config.mean(axis, dtype, out, keepdims, *args, **kwargs)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *args, **kwargs):
        return self._config.var(axis, dtype, out, ddof, keepdims, *args, **kwargs)

    def pformat(self, delim: str = " ") -> str:
        """Returns a formated string of the configuration."""
        header = r"i\l  " + delim.join([f"{i:^3}" for i in range(self.num_timesteps)])
        rows = list()
        for site in range(self.num_sites):
            row = delim.join([f"{x:^3}" for x in self._config[site, :]])
            rows.append(f"{site:<3} [{row}]")
        return header + "\n" + "\n".join(rows)

    def __getitem__(self, item):
        return self._config[item]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape: {self.shape}, dtype={self._config.dtype})"

    def __str__(self) -> str:
        return self.pformat()
