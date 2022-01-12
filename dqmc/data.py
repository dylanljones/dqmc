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

    def get_results(self, kwargs):
        group = self.get_simulation_group(kwargs)
        if group is None or len(group) == 0:
            return list()
        results = [np.array(group[k]) for k in group]
        return results

    def save_results(self, kwargs, results):
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


def compute_dataset(data, p):
    results = data.get_results(p)
    if not results:
        results = run_dqmc(p)
        data.save_results(p, results)
    return results


def compute_datasets(data, params, max_workers=None, batch=None, callback=None,
                     progress=True, overwrite=False):
    # Check which datasets allready exist
    com_params = list()
    for p in params:
        res = data.get_results(p)
        if overwrite or not res:
            com_params.append(p)
    # Compute missing datasets in parallel
    if com_params:
        logger.info("Computing %s datasets", len(com_params))
        max_workers = get_max_workers(max_workers)

        if batch is None or batch == 0:
            batch = len(com_params)
        for i in range(0, len(com_params), batch):
            batch_params = com_params[i:i + batch]
            results = run_dqmc_parallel(batch_params, callback, max_workers, progress)
            for p, res in zip(batch_params, results):
                data.save_results(p, res)

    # Get all results
    results = list()
    for p in params:
        res = data.get_results(p)
        results.append(res)
    return results
