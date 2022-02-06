# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones

import os
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import gftool as gt
from dqmc import Parameters, transpose_results
from dqmc import map_params, Database, update_datasets

logger = logging.getLogger("dqmc")
logger.setLevel(logging.WARNING)


def timeit(func, *args, repreat=1000):
    times = list()
    for _ in range(repreat):
        t0 = time.perf_counter()
        func(*args)
        t = time.perf_counter() - t0
        times.append(t)
    fname = func.__name__
    tavrg = np.mean(times)
    unit = "s"
    print(f"{fname:<30} {tavrg:.2e} {unit}")


def save_figure(arg, *relpaths, dpi=600, frmt=None, rasterized=True):
    if not isinstance(arg, plt.Figure):
        fig = arg.get_figure()
    else:
        fig = arg
    print(f"Saving...", end="", flush=True)
    if rasterized:
        for ax in fig.get_axes():
            ax.set_rasterized(True)
    file = os.path.join(*relpaths)
    if (frmt is not None) and (not file.endswith(frmt)):
        filename, _ = os.path.splitext(file)
        file = filename + "." + frmt
    fig.savefig(file, dpi=dpi, format=frmt)
    print(f"\rFigure saved: {os.path.split(file)[1]}")
    return file


def gf_tau2iw(gf_tau, beta, n_points=None):
    r"""Transforms the real time Green's function to the Matsubara domain.

    Parameters
    ----------
    gf_tau : (..., N) np.ndarray
        The Green's function at imaginary times :math:`τ \in [0, β]`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.
    n_points : int array_like, optional
        Points for which the Matsubara frequencies :math:`iω_n` are generated.
        If ``None`` the default number of ``N/2`` is used.

    Returns
    -------
    iw : complex np.ndarray
        Array of the imaginary Matsubara frequencies.
    gf_iw : (..., (M + 1)/2) complex np.ndarray
        The Fourier transform of `gf_tau` for non-negative fermionic Matsubara
        frequencies :math:`iω_n`.
    """
    if n_points is None:
        num_times = gf_tau.shape[0]
        n_points = range(num_times // 2)
    iws = gt.matsubara_frequencies(n_points, beta=beta)
    gf_iw = gt.fourier.tau2iw(gf_tau, beta=beta)
    return iws, gf_iw


def average_gf(gf, axis1=-2, axis2=-1):
    return np.trace(gf, axis1=axis1, axis2=axis2) / gf.shape[axis1]


def plot_gftau(tau, gftau_up, gftau_dn, ax=None, file="", yoffset=(0.02, 0.02)):
    gf_avrg_up = average_gf(gftau_up)
    gf_avrg_dn = average_gf(gftau_dn)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.plot(tau, gf_avrg_up, marker="o", ms=4, label="G_up", ls="-")
    ax.plot(tau, gf_avrg_dn, marker="o", ms=4, label="G_dn", ls="-")
    ax.legend()
    ax.grid()
    ax.set_xlabel(r"$\tau$")
    ax.set_xlim(tau[0] - 0.1, tau[-1] + 0.1)
    ax.set_ylim(-yoffset[0], 0.5 + yoffset[1])

    ax.set_ylabel(r"G$(\tau, 0)$")
    if file:
        fig.savefig(file)
    return ax


def plot_gftau_u(tau, gfs, u_values, file="", yoffset=(0.02, 0.02), ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for u, gf in zip(u_values, gfs):
        ax.plot(tau, average_gf(gf), marker="o", ms=4, label=f"U={u}")
    ax.legend()
    ax.grid()
    ax.set_xlabel(r"$\tau$")
    ax.set_xlim(tau[0] - 0.1, tau[-1] + 0.1)
    ax.set_ylabel(r"G$(\tau, 0)$")
    ax.set_ylim(-yoffset[0], 0.5 + yoffset[1])

    if file:
        fig.savefig(file)
    return ax


def plot_gftau0_contourf(ax, x, y, gf, levels=10, clip=1.0, cmap=None, colorbar=True):
    xx, yy = np.meshgrid(x, y)
    zz = np.abs(gf)
    vmin, vmax = None, None
    if clip is not None:
        vmin = 0.0
        vmax = clip
        zz = np.clip(zz, vmin+1e-4, vmax-1e-4)
    cont = ax.contourf(xx, yy, zz, levels, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        fig = ax.get_figure()
        cbar = fig.colorbar(cont)
        cbar.ax.set_ylabel(r"G$(\tau, 0)$")
        if clip and np.max(zz) > 0.8:
            labels = [item.get_text() for item in cbar.ax.get_yticklabels()]
            labels[-1] = f">{labels[-1]}"
            cbar.ax.set_yticklabels(labels)
    else:
        cbar = None

    return cont, cbar


def plot_gftau0_u_contour(dt, u_values, gftau0, levels=10, clip=1, cmap="bwr", ax=None):
    gfs = average_gf(gftau0)
    tau = np.arange(len(gfs[0])) * dt

    if ax is None:
        fig, ax = plt.subplots()
    plot_gftau0_contourf(ax, tau, u_values, gfs, levels, clip, cmap)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$U$")
    ax.set_title(rf"$\Delta \tau={dt}$ ($L={len(tau)}$)")
    return ax


def compute_gftau(db, params, workers=4, batch=None):
    results = update_datasets(db, params, max_workers=workers, batch_size=batch)
    results = transpose_results(results)
    return results[-3], results[-2]


def plot_results(db, p, u_plot, u_values, gftau0_up):
    dt = p.dt
    ax = plot_gftau0_u_contour(dt, u_values, gftau0_up, levels=15)
    for y in u_plot:
        ax.axhline(y=y, ls="--", color="k", lw=1)

    fig, ax1 = plt.subplots()
    ax1.grid()
    ax1.set_xlabel(r"$\tau$")
    ax1.set_ylabel(r"$G(\tau)$")
    ax1.set_ylim(0, 0.52)
    ax1.set_title(rf"$\Delta \tau={dt}$ ($L={p.num_times}$)")

    fig, ax2 = plt.subplots()
    ax2.set_xlabel(r"$i\omega$")
    ax2.set_ylabel(r"$G(i\omega)$")
    ax2.set_ylim(0.0, 0.4)
    ax2.grid()
    ax2.set_title(rf"$\Delta \tau={dt}$ ($L={p.num_times}$)")

    for u in u_plot:
        p.u = float(u)
        results = db.get_results(p)

        gftau_up = average_gf(results[-3])
        gftau_dn = average_gf(results[-2])
        tau = np.arange(len(gftau_up)) * dt
        ax1.plot(tau, gftau_up, label=f"U={p.u}")

        beta = p.beta - p.dt
        iws, gfiw_up = gf_tau2iw(gftau_up, beta)
        line = ax2.plot(iws.imag, gfiw_up.imag, label=f"U={p.u}")[0]
        # ax2.plot(iws.imag, gfiw_up.real, color=line.get_color(), ls="--")

    ax1.legend()
    ax2.legend()


def load_gftau_arrays(db, p, u_values):
    if isinstance(u_values, (int, float)):
        p.u = float(u_values)
        results = db.get_results(p)
    else:
        params = map_params(p, u=u_values)
        results = db.get_results(*params)
        results = transpose_results(results)
    return results[-3], results[-2]


def main():
    db = Database("gftau0.hdf5")
    n = 10
    beta = 4.0
    dt = 0.02
    nt = int(beta / dt)
    p = Parameters(n, dt=dt, num_times=nt, prod_len=8, num_wraps=8, num_sampl=2*2048)
    u_values = np.arange(0.1, 10, 0.1)
    params = map_params(p, u=u_values)
    gftau0_up, gftau0_dn = compute_gftau(db, params, workers=4, batch=8)

    u_plot = [2, 4, 5, 6, 8]

    plot_results(db, p, u_plot, u_values, gftau0_up)
    plt.show()


if __name__ == "__main__":
    main()
