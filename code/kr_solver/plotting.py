"""Plotting utilities for the KR gravity solver results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .yukawa import YukawaFit, yukawa_potential


def plot_potential(r: np.ndarray, phi: np.ndarray, outfile: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, phi, color="navy", label=r"$\Phi(r)$")
    ax.set_xlabel("r (AU)")
    ax.set_ylabel(r"$\Phi(r)$")
    ax.set_title("Gravitational Potential")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_kr_fields(
    r: np.ndarray, b01: np.ndarray, b23: np.ndarray, outfile: Path | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, b01, color="crimson", label=r"$b_{01}(r)$")
    ax.plot(r, b23, color="seagreen", label=r"$b_{23}(r)$")
    ax.set_xlabel("r (AU)")
    ax.set_ylabel("Field amplitude")
    ax.set_title("Kalb--Ramond field components")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_yukawa_fit(
    r: np.ndarray, phi: np.ndarray, fit: YukawaFit, outfile: Path | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, phi, color="navy", label="Numerical $\\Phi(r)$")
    ax.plot(
        r,
        yukawa_potential(r, fit.A, fit.mu),
        color="darkorange",
        linestyle="--",
        label=fr"Yukawa fit: $A={fit.A:.3f}$, $\mu={fit.mu:.3e}$",
    )
    ax.set_xlabel("r (AU)")
    ax.set_ylabel(r"$\Phi(r)$")
    ax.set_title("Yukawa fit to numerical potential")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_convergence(hist: dict, outfile: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, series in hist.items():
        ax.semilogy(series, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative change")
    ax.set_title("Convergence of coupled fields")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200)
    plt.close(fig)
