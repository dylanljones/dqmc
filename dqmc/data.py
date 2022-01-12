# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import h5py
import hashlib
import logging
import numpy as np
from typing import Union
from .simulator import Parameters, run_dqmc
from .mp import get_max_workers, run_dqmc_parallel

logger = logging.getLogger("dqmc")


def hash_params(**kwargs):
    keys = sorted(kwargs.keys())
    data = "; ".join([str(kwargs[k]) for k in keys])
    m = hashlib.md5(data.encode("utf-8"))
    return m.hexdigest()


def check_attrs(item: Union[h5py.File, h5py.Group, h5py.Dataset],
                attrs: dict, mode: str = "equals") -> bool:
    """Checks if the attributes of an hdf5-object match the given attributes.

    Parameters
    ----------
    item : h5py.File or h5py.Group or h5py.Database
        The attributes of the item are checked.
    attrs : dict
        The attributes for matching.
    mode : str, optional
        Mode for matching attributes. Valid modes are 'equals' and 'contains'.
        If the mode is 'equals', the dictionary of the item attributes has to be
        equal to the giben dictionary. If the mode is 'contains', the item dictionary
        can contain any number of values, but the values of the given dictionary have
        to be included. The default is 'equals'.

    Returns
    -------
    matches: bool
    """
    if mode == "contains":
        for key, val in attrs.items():
            if key not in item.attrs.keys() or item.attrs[key] != val:
                return False
        return True
    elif mode == "equals":
        return item.attrs == attrs
    else:
        modes = ["contains", "equals"]
        raise ValueError(f"Mode '{mode}' not supported! Valid modes: {modes}")


class Database:
    """HDF5 database for storing DQMC simulation results.

    Uses the hash of the input parameters (`dict` or `Parameters`) as key to store
    the results of a DQMC simulation.
    """

    def __init__(self, file="dqmc.hdf5", mode="a"):
        self.file = file if isinstance(file, h5py.File) else h5py.File(file, mode)

    @staticmethod
    def get_group_name(kwargs: Union[dict, Parameters]):
        if isinstance(kwargs, Parameters):
            kwargs = kwargs.__dict__
        return str(hash_params(**kwargs))

    def get_simulation_group(self, kwargs):
        name = self.get_group_name(kwargs)
        group = self.file.get(name, default=None)
        return group

    def create_simulation_group(self, kwargs):
        name = self.get_group_name(kwargs)
        group = self.file.get(name, default=None)
        if group is None:
            group = self.file.create_group(name, track_order=True)
        return group

    def get_groups(self):
        return [self.file[k] for k in self.file]

    def find_groups(self, attrs, mode="contains"):
        groups = list()
        for k in self.file.keys():
            item = self.file[k]
            if check_attrs(item, attrs, mode):
                groups.append(item)
        return groups

    def find_missing(self, params, overwrite=False):
        if overwrite:
            return list(params)
        missing_params = list()
        for p in params:
            res = self.get_results(p)
            if not res:
                missing_params.append(p)
        return missing_params

    def get_results(self, *params):
        results = list()
        for param in params:
            group = self.get_simulation_group(param)
            if group is None or len(group) == 0:
                return list()
            res = [np.array(group[k]) for k in group]
            results.append(res)
        return results[0] if len(params) == 1 else results

    def save_results(self, kwargs: Union[dict, Parameters], results):
        if isinstance(kwargs, Parameters):
            kwargs = kwargs.__dict__
        group = self.create_simulation_group(kwargs)
        # Create datasets
        keys = ["gf_up", "gf_dn", "n_up", "n_dn", "n_dbl", "moment", "user"]
        for k, res in zip(keys, results):
            if k in group:
                del group[k]
            group.create_dataset(k, data=res)

        # Update attributes of group
        for k, v in kwargs.items():
            group.attrs[k] = v
        return group

    def treestr(self):
        string = f"{self.file}\n"
        for key in self.file:
            group = self.file[key]
            string += f"   {group}\n"
            for k in group:
                string += f"      {group[k]}\n"
        return string


def compute_dataset(db, p):
    results = db.get_results(p)
    if not results:
        results = run_dqmc(p)
        db.save_results(p, results)
    return results


def compute_datasets(db, params, max_workers=None, batch_size=None, callback=None,
                     progress=True, header=None):
    """Runs multiple DQMC simulations in parallel and stores the results in a database.

    Parameters
    ----------
    db : Database
        The database instance used to store the DQMC simulation results.
    params : Iterable of Parameters
        The input parameters to map to the processes. The hash of the parameters
        are used as a key to store the results of the DQMC simulation in the database.
        The list of the returned results preserves the input order of the parameters.
    callback : callable, optional
        A optional callback method for measuring additional observables.
    max_workers : int, optional
        The number of processes to use (see `get_max_workers`). If `None` or `0`
        the number of logical cores of the system is used. If a negative integer
        is passed it is subtracted from the number of logical cores. For example,
        `max_workers=-1` uses the number of cores minus one as number of processes.
    batch_size : int, optional
        The number of results that are computed in one batch before saving the
        results to the database. Should not be less than the number of processes used.
        If `None` all simulations are run in a single batch, if `0` the batch
        size is set to the number of processes.
    progress : bool, optional
        If `True` a progresss bar is printed.
    header : str, optional
        A header for printing the progress bar. Ignored if `progress=False`.
    Returns
    -------
    results : List
        The results of the DQMC simulations in the order of the input parameters.
    """
    if not params:
        return

    # Get number of processes here to use for calculation of batch size
    max_workers = get_max_workers(max_workers)
    if batch_size is None:
        batch_size = len(params)
    elif batch_size == 0:
        batch_size = max_workers

    # Warn if batch size to small
    if batch_size < max_workers:
        logger.warning("Batch size `%s` is lower than the number of processes `%s`",
                       batch_size, max_workers)

    # Run simulations and store after each batch
    num_params = len(params)
    num_batches = int(np.ceil(num_params / batch_size))
    desc = None
    for batch, i in enumerate(range(0, len(params), batch_size)):
        # Get params of batch
        batch_params = params[i:i + batch_size]
        # Format header if given
        if header is not None:
            desc = header
            if batch_size != num_params:
                desc += f" {batch + 1}/{num_batches}"
        # Run DQMC simulations
        results = run_dqmc_parallel(batch_params, callback, max_workers, progress, desc)
        # Save results to database
        for p, res in zip(batch_params, results):
            db.save_results(p, res)

    # Get all results
    return db.get_results(*params)


def update_datasets(db, params, max_workers=None, batch_size=None, callback=None,
                    overwrite=False, progress=True, header=None):
    """Updates the database to contain all results of the passed simulation parameters.

    Checks which datasets are missing in the database and computes the missing results
    of DQMC simulations in parallel and stores them in the database.

    Parameters
    ----------
    db : Database
        The database instance used to store the DQMC simulation results.
    params : Iterable of Parameters
        The input parameters to map to the processes. The hash of the parameters
        are used as a key to store the results of the DQMC simulation in the database.
        The list of the returned results preserves the input order of the parameters.
    callback : callable, optional
        A optional callback method for measuring additional observables.
    max_workers : int, optional
        The number of processes to use (see `get_max_workers`). If `None` or `0`
        the number of logical cores of the system is used. If a negative integer
        is passed it is subtracted from the number of logical cores. For example,
        `max_workers=-1` uses the number of cores minus one as number of processes.
    batch_size : int, optional
        The number of results that are computed in one batch before saving the
        results to the database. Should not be less than the number of processes used.
        If `None` or `0` all simulations are run in a single batch, if `1` the batch
        size is set to the number of processes.
    overwrite : bool, optional
        If `True`, existing datasets are overwritten and all simulations are run.
    progress : bool, optional
        If `True` a progresss bar is printed.
    header : str, optional
        A header for printing the progress bar. Ignored if `progress=False`.
    Returns
    -------
    results : List
        The results of the DQMC simulations in the order of the input parameters.

    See Also
    --------
    compute_datasets : Runs DQMC simulations and stores the results in the database.
    """
    # Check which datasets allready exist
    missing = db.find_missing(params, overwrite)
    # Compute missing datasets and store in database
    if missing:
        compute_datasets(db, missing, max_workers, batch_size, callback, progress,
                         header)
    # Get all results (existing and new) from the database
    return db.get_results(*params)
