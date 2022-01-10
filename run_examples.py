# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones

import os.path
import numpy as np
import matplotlib.pyplot as plt
from dqmc import map_params, run_dqmc_parallel, parse


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


def average_results(results):
    tresults = transform_results(results)
    for i in range(len(tresults) - 1):
        tresults[i] = np.mean(tresults[i], axis=1)
    return tresults


def plot_local_moment(file, temps, interactions, max_workers=-1, save=True):
    p_default = parse(file)
    if not hasattr(interactions, "__len__"):
        interactions = [interactions]

    directory, filename = os.path.split(file)
    name = os.path.splitext(filename)[0]
    figpath = os.path.join(directory, name + "_moment" + ".png")

    fig, ax = plt.subplots()
    # ax.set_title("Input: " + filename)
    ax.set_xscale("log")
    for u in interactions:
        p = p_default.copy(u=u)
        params = map_params(p, temp=temps)
        # Run simulations and build results
        results = run_dqmc_parallel(params, max_workers=max_workers)
        n_up, n_dn, n_double, moment, _ = average_results(results)
        # Plot local moment
        ax.plot(temps, moment, marker="o", ms=4, label=f"$U={u}$")

    ax.legend()
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\langle m_z^2 \rangle$")
    ax.grid()
    # fig.tight_layout()
    if save:
        fig.savefig(figpath)

    return fig, ax


def main():
    root = "examples"
    temps = np.geomspace(0.1, 100, 20)
    inter = [1, 2, 4, 6, 8]

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
        print(f"Running simulations for {file}")
        plot_local_moment(file, temps, inter)
        print()


if __name__ == "__main__":
    main()
