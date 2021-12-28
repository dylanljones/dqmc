# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Multiprocessing tools."""

import logging
import psutil
import numpy as np  # noqa: F401
import concurrent.futures
from tqdm import tqdm
from .simulator import run_dqmc, Parameters

logger = logging.getLogger("dqmc")


# noinspection PyShadowingNames
def map_params(p, **kwargs):
    """Maps arrays to the attributes of a default Parameter object.

    Parameters
    ----------
    p : Parameters
        The default parameters. A copy of the parameters is created
        before replacing the attributes from the keyword arguments
    **kwargs
        Keyword arguments containing arrays of values that are mapped to the
        attributes of the default parameters. If multiple keyword arguments
        are given the length of all arrays have to match.
    Returns
    -------
    params : list of Parameters
        The parameters with the mapped keyword arguments.

    Examples
    --------
    >>> p = Parameters(10, 4, 0, 1, 2, 0.05, 40)  # Default parameters
    >>> u = np.arange(1, 3, 0.5)  # Interactions
    >>> params = map_params(p, u=u)  # Map interaction array to parameters
    >>> [p.u for p in params]  # Interaction has mapped values
    [1.0, 1.5, 2.0, 2.5]
    >>> [p.t for p in params]  # Hopping is constant for all parameters
    [1, 1, 1, 1]
    """
    num_params = 0
    # Check number of values for each keyword argument
    for key, vals in kwargs.items():
        num_vals = len(vals)
        if num_params == 0:
            num_params = num_vals
        elif num_vals != num_params:
            raise ValueError(f"Length {num_vals} of keyword argument {key} does not "
                             f"match the previous lengths {num_params}")
    # Map parameters
    params = list()
    for i in range(num_params):
        # Copy default parameters
        p_new = Parameters(**p.__dict__)
        # Update new parameters with given kwargs
        for key, vals in kwargs.items():
            val = vals[i]
            setattr(p_new, key, val)
        params.append(p_new)
    return params


def run_dqmc_parallel(params, callback=None, max_workers=None, progress=True):
    if max_workers is None or max_workers == -1:
        max_workers = psutil.cpu_count(logical=True)

    args = [(p, callback) for p in params]
    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
        results = executor.map(run_dqmc, *zip(*args))
        if progress:
            return list(tqdm(results, total=len(args)))
        else:
            return list(results)
