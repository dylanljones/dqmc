# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import os.path
import numpy as np
import matplotlib.pyplot as plt
from dqmc import map_params, parse
from dqmc.data import Database, compute_datasets


def transform_results(results):
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


def average_observables(results):
    # Only use observables, not Green's functions (first two items)
    tresults = transform_results(results)[2:]
    # Average observables
    for i in range(len(tresults) - 1):
        tresults[i] = np.mean(tresults[i], axis=1)
    return tresults


def compute_data_temp(db, file, temps, interactions, max_workers=-1, batch=None,
                      overwrite=False):

    print(f"Running simulations for {file}")
    p_default = parse(file)
    for u in interactions:
        p = p_default.copy(u=u)
        params = map_params(p, temp=temps)

        # Check which datasets allready exist
        missing = db.find_missing(params, overwrite)
        # Compute missing datasets and store in database
        head = f"U={u}"
        if missing:
            compute_datasets(db, missing, max_workers, batch_size=batch, header=head)
        else:
            print(f"{head}: Found existing data!")
    print("Done!\n")


def plot_local_moment(db, file, temps, interactions, save=True):
    p_default = parse(file)

    directory, filename = os.path.split(file)
    name = os.path.splitext(filename)[0]
    figpath = os.path.join(directory, name + "_moment" + ".png")

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    for u in interactions:
        p = p_default.copy(u=u)
        params = map_params(p, temp=temps)
        results = db.get_results(*params)
        n_up, n_dn, n_double, moment, _ = average_observables(results)
        ax.plot(temps, moment, marker="o", ms=3, label=f"$U={u}$")

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\langle m_z^2 \rangle$")
    ax.set_ylim(0.48, 1.02)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.grid()
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(figpath)
    return fig, ax


def plot_magnetization(db, file, temps, interactions, save=True):
    p_default = parse(file)

    directory, filename = os.path.split(file)
    name = os.path.splitext(filename)[0]
    figpath = os.path.join(directory, name + "_mag" + ".png")

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    for u in interactions:
        p = p_default.copy(u=u)
        params = map_params(p, temp=temps)
        results = db.get_results(*params)
        n_up, n_dn, n_double, moment, _ = average_observables(results)
        mag = n_up - n_dn
        ax.plot(temps, mag, marker="o", ms=3, label=f"$U={u}$")

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\langle m_z \rangle$")
    # ax.set_ylim(0.48, 1.02)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.grid()
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(figpath)
    return fig, ax


def main():
    root = "examples"
    batch = None
    overwrite = False

    db = Database(os.path.join(root, "examples.hdf5"))

    temps = np.geomspace(0.1, 100, 20)
    inter = [1, 2, 4, 6, 8]
    max_workers = -1

    # Find all text files in `root` directory
    files = list()
    for name in os.listdir(root):
        if os.path.splitext(name)[1] == ".txt":
            path = os.path.join(root, name)
            files.append(path)
    print(f"Found {len(files)} input files!")
    print()

    # Run simulations for each input file
    for file in files:
        compute_data_temp(db, file, temps, inter, max_workers, batch, overwrite)
        plot_local_moment(db, file, temps, inter)


if __name__ == "__main__":
    main()
