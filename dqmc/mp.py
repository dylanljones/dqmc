# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
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


def get_max_workers(max_workers=None):
    """Returns the number of processes used to run multiple workloads in parallel.

    Parameters
    ----------
    max_workers : int, optional
        The number of processes to use. If `None` or `0` the number of
        logical cores of the system is used. If a negative integer is passed it
        is subtracted from the number of logical cores. For example, `max_workers=-1`
        uses the number of cores minus one as number of processes.
    Returns
    -------
    max_workers : int
        The number of processes to use.
    """
    if max_workers is None or max_workers == 0:
        max_workers = psutil.cpu_count(logical=True)
    elif max_workers < 0:
        max_workers = max(1, psutil.cpu_count(logical=True) - abs(max_workers))
    return max_workers


def transpose_results(results):
    """Transposes a list of arrays."""
    tresults = [list() for _ in range(len(results[0]))]
    num_res = len(tresults)
    for res in results:
        for i in range(num_res):
            tresults[i].append(res[i])

    # Last result is user callback, so not guaranted to be convertible to np.array
    out = [np.array(x) for x in tresults[:-1]]
    last = tresults[-1]
    try:
        last = np.array(last)
    except Exception:  # noqa
        pass
    out.append(last)

    return out


# noinspection PyShadowingNames
def run_dqmc_parallel(params, unequal_time=True, callback=None, max_workers=None,
                      progress=True, header=None, transpose=False):
    """Runs multiple DQMC simulations in parallel.

    Parameters
    ----------
    params : Iterable of Parameters
        The input parameters to map to the processes. The list of results preserves
        the input order of the parameters.
    unequal_time : bool
        If `True`, unequal-time measurements will be performed.
    callback : callable, optional
        A optional callback method for measuring additional observables.
    max_workers : int, optional
        The number of processes to use (see `get_max_workers`). If `None` or `0`
        the number of logical cores of the system is used. If a negative integer
        is passed it is subtracted from the number of logical cores. For example,
        `max_workers=-1` uses the number of cores minus one as number of processes.
    progress : bool, optional
        If `True` a progresss bar is printed.
    header : str, optional
        A header for printing the progress bar. Ignored if `progress=False`.
    transpose : bool, optional
        If `True`, the list of results is transposed and cast to `np.ndarray`, such
        that the first index corresponds to the observable.

    Returns
    -------
    results : List
        The results of the DQMC simulations in the order of the input parameters.

    See Also
    --------
    get_max_workers : Determines the number of processes used.

    Examples
    --------
    >>> p = Parameters(10, 4, 0, 1, 2, 0.05, 40)  # Default parameters
    >>> u = np.arange(1, 3, 0.5)  # Interactions
    >>> params = map_params(p, u=u)  # Map interaction array to parameters
    >>> res = run_dqmc_parallel(params, max_workers=-1)
    """
    max_workers = get_max_workers(max_workers)

    args = [(p, unequal_time, callback) for p in params]
    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
        results = executor.map(run_dqmc, *zip(*args))
        if progress:
            results = list(tqdm(results, total=len(args), desc=header))
        else:
            results = list(results)

    return transpose_results(results) if transpose else results
