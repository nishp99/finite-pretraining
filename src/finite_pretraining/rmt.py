"""Population spectral window, conditional overlap profile and nine weighted traces."""

import numpy as np
import jax
from ._numerics.spectrum import find_window, interval_grid, densities, validate_width
from ._numerics.traces import (
    WeightedTraces,
    spectral_weight,
    encoder_weighted_traces,
    compute_weighted_traces_diagnostic,
)


def trace_statistics(
    alpha, pretraining_time, *, phi=0.2, rho=0.8, delta=0.1, q0=0.05, kappa=0.0
):
    """Compute D-normalized moments with doubled-grid and cubic contour error checks."""
    options = dict(
        quadrature="tensor",
        n_grid1d=1024,
        n_grid2=256,
        tensor_nodes=(64, 128, 256, 512, 1024),
        eta_fractions=(0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.000003),
        atol=1e-5,
        rtol=1e-3,
    )
    t = float(pretraining_time)
    jax.config.update("jax_enable_x64", True)
    with jax.default_device(jax.devices("cpu")[0]):
        r = compute_weighted_traces_diagnostic(
            alpha,
            rho,
            delta,
            phi=phi,
            ell=t,
            q0=q0,
            kappa=kappa,
            integration_options=options,
        )
    if kappa == 0 and (t == 0 or np.isposinf(t)):
        c = q0 if t == 0 else 1.0
        v = np.array(r["values"])
        r["unreduced_values"] = v.copy()
        v[:3] = [phi * c**i for i in (1, 2, 3)]
        v[4:6] = [c * v[3], c * c * v[3]]
        v[7] = c * v[6]
        r["values"] = WeightedTraces(*v)
    return r


def conditional_profile(nodes, alpha, rho=0.8, delta=0.1):
    density, signal = densities(nodes, alpha, rho, delta)
    p = (1 - rho) * signal / density
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ArithmeticError("Invalid RMT conditional profile")
    return p


def calibration_grid(alpha, phi, rho, delta):
    window = find_window(alpha, rho, delta, phi=phi)
    x, w, _ = interval_grid(window["intervals"], 1024)
    density, _ = densities(x, alpha, rho, delta)
    order = np.argsort(x)
    mass = (w * density)[order]
    cdf = (np.cumsum(mass) - mass / 2) / mass.sum()
    return np.interp((np.arange(4096) + 0.5) / 4096, cdf, x[order]), window["intervals"]
