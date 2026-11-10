"""Actual aligned, tied LAE encoders. Columns are signal coordinates, then noise."""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from .rmt import spectral_weight, validate_width


@dataclass(frozen=True)
class Dataset:
    samples: np.ndarray
    signal_dim: int
    delta: float


@dataclass(frozen=True)
class Encoder:
    weights: np.ndarray
    signal_dim: int
    delta: float
    pretraining_time: float
    method: str


def _count(value, name):
    n = round(value)
    if n < 1 or not np.isclose(n, value, atol=1e-9, rtol=0):
        raise ValueError(f"{name} must be an integer; specify the realized fraction")
    return n


def generate_dataset(D=200, alpha=0.75, *, rho=0.8, delta=0.1, seed=0):
    """Draw N=alpha*D independent Gaussian samples with two population variances."""
    if (
        not isinstance(D, int)
        or D < 2
        or not (0 < rho < 1 and 0 < delta < 1 and np.isfinite(alpha) and alpha > 0)
    ):
        raise ValueError("Invalid dataset parameters")
    n = _count(D * alpha, "alpha*D")
    s = _count(D * (1 - rho), "(1-rho)*D")
    x = np.random.default_rng(seed).normal(size=(n, D))
    x[:, s:] *= np.sqrt(delta)
    return Dataset(x, s, delta)


def _principal(data, phi):
    x = data.samples
    n, d = x.shape
    validate_width(n / d, 1 - data.signal_dim / d, data.delta, phi)
    k = _count(phi * d, "phi*D")
    cov = x.T @ x / n
    ev, u = eigh(cov, subset_by_index=(d - k, d - 1))
    return cov, ev, u.T


def spectral_encoder(data, pretraining_time, *, phi=0.2, q0=0.05, kappa=0.0):
    """Analytic aligned flow on this dataset's own sample covariance."""
    _, ev, axes = _principal(data, phi)
    q = spectral_weight(ev, float(pretraining_time), q0, kappa)
    return Encoder(
        np.sqrt(q)[:, None] * axes,
        data.signal_dim,
        data.delta,
        float(pretraining_time),
        "spectral",
    )


def train_lae(data, pretraining_time, *, phi=0.2, q0=0.05, kappa=0.0, steps=500):
    """Full matrix GD on mean squared reconstruction loss + (2*kappa/D)||E||².

    Uses constant eta such that ell=4*sum(eta)/D. Returns the learned matrix,
    not its analytic spectral approximation. Increase steps to refine GD.
    """
    t = float(pretraining_time)
    if (
        not np.isfinite(t)
        or t < 0
        or not isinstance(steps, int)
        or steps < 1
        or not (0 < q0 < 1 and np.isfinite(kappa) and kappa >= 0)
    ):
        raise ValueError(
            "GD needs finite nonnegative time, valid q0/kappa and positive integer steps"
        )
    cov, _, axes = _principal(data, phi)
    d = cov.shape[0]
    w = np.sqrt(q0) * axes
    eta = t * d / (4 * steps)
    for _ in range(steps):
        wc = w @ cov
        grad = (-4 * wc + 2 * (w @ w.T) @ wc + 2 * (wc @ w.T) @ w + 4 * kappa * w) / d
        w = w - eta * grad
        if not np.isfinite(w).all():
            raise ArithmeticError(
                "GD diverged; increase steps at fixed pretraining time"
            )
    return Encoder(w, data.signal_dim, data.delta, 4 * steps * eta / d, "aligned_gd")
