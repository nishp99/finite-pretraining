"""Finite-time tied-LAE traces. rho is noise fraction; phi is hidden width.

Host orchestration and float64 JAX kernels follow the projector provider.
These constants do not make the existing projector ODEs finite-time ODEs.
"""

from typing import NamedTuple
import math
import time
import numpy as np
from .spectrum import validate_width, find_window, interval_grid, densities


class WeightedTraces(NamedTuple):
    m1: float
    m2: float
    m3: float
    s1: float
    s2: float
    s3: float
    T_ss: float
    U: float
    tau_sss: float


def _validate_flow(ell, q0, kappa):
    if not (
        0 < q0 < 1
        and np.isfinite(q0)
        and ell >= 0
        and not np.isnan(ell)
        and np.isfinite(kappa)
        and kappa >= 0
    ):
        raise ValueError("Require ell>=0 (possibly infinity), 0<q0<1, kappa>=0.")


def spectral_weight(x, ell=np.inf, q0=0.05, kappa=0.0):
    """Stable real solution and analytic continuation of dq/dell=2q[x(1-q)-kappa].

    At infinite time, complex arguments describe the positive active spectrum;
    integration callers split the real window at kappa first. For kappa=0
    this analytic continuation is identically one (including Re(x)=0).
    """
    _validate_flow(ell, q0, kappa)
    x = np.asarray(x)
    if np.any(~np.isfinite(x)) or np.any(x.real < 0):
        raise ValueError(
            "Spectral arguments must be finite with nonnegative real part."
        )
    dtype = np.complex128 if np.iscomplexobj(x) else np.float64
    x = x.astype(dtype)
    if ell == 0:
        return np.full(x.shape, q0, dtype=dtype)
    if np.isinf(ell):
        if np.iscomplexobj(x):
            if kappa == 0:
                return np.ones_like(x)
            if np.any(x.real < kappa):
                raise ValueError("Infinite-time analytic weight requires Re(x)>=kappa.")
            return 1 - kappa / x
        out = np.zeros_like(x)
        positive = x > kappa
        np.divide(x - kappa, x, out=out, where=positive)
        # A zero empirical mode has no gradient when kappa=0.
        return np.where((x == 0) & (kappa == 0), q0, out)
    a = x - kappa
    z = 2 * a * ell
    # Two algebraically identical forms avoid overflow in either half-plane.
    near = np.abs(z) < 1e-5
    ap = np.where(near, 1.0, a)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eneg = (
            np.exp(-np.maximum(z.real, 0) - 1j * z.imag)
            if np.iscomplexobj(z)
            else np.exp(-np.maximum(z, 0))
        )
        epos = (
            np.exp(np.minimum(z.real, 0) + 1j * z.imag)
            if np.iscomplexobj(z)
            else np.exp(np.minimum(z, 0))
        )
        upper = q0 / (eneg + x * q0 * (-np.expm1(-np.where(z.real >= 0, z, 0))) / ap)
        lower = q0 * epos / (1 + x * q0 * np.expm1(np.where(z.real < 0, z, 0)) / ap)
        # expm1(z)/z is analytic at zero; retain enough terms for float64.
        exprel = 1 + z / 2 + z * z / 6 + z**3 / 24 + z**4 / 120
        local = q0 * np.exp(np.where(near, z, 0)) / (1 + 2 * x * q0 * ell * exprel)
    return np.where(near, local, np.where(z.real >= 0, upper, lower))


def encoder_weighted_traces(encoder, signal_dim):
    """All nine D-normalized traces of actual E, without orthonormalization."""
    e = np.asarray(encoder, dtype=np.float64)
    if e.ndim != 2 or not 0 < signal_dim <= e.shape[1] or not np.isfinite(e).all():
        raise ValueError("Expected a finite matrix and valid signal dimension.")
    h, b = e @ e.T, e[:, :signal_dim] @ e[:, :signal_dim].T
    h2, b2 = h @ h, b @ b
    d = e.shape[1]
    return WeightedTraces(
        *(
            float(v) / d
            for v in (
                np.trace(h),
                np.sum(h * h),
                np.sum(h2 * h),
                np.trace(b),
                np.sum(h * b),
                np.sum(h2 * b),
                np.sum(b * b),
                np.sum(h * b2),
                np.sum(b2 * b),
            )
        )
    )


def _active_window(window, ell, kappa):
    if not (np.isinf(ell) and kappa > 0):
        return window
    intervals = tuple(
        (max(lo, kappa), hi) for lo, hi in window["intervals"] if hi > kappa
    )
    return dict(
        window,
        intervals=intervals,
        lambda_cut=intervals[0][0] if intervals else window["lambda_edge"],
    )


def _lower_traces(alpha, rho, Delta, window, ell, q0, kappa, n1, n2):
    import jax.numpy as jnp
    from .boundary import _boundary_data, _two_point_boundary_kernels

    xs, ws, _ = interval_grid(window["intervals"], n1)
    rg, rs = densities(xs, alpha, rho, Delta)
    q = spectral_weight(xs, ell, q0, kappa)
    powers = q[:, None] ** np.arange(1, 4)
    m, s = (ws * rg) @ powers, (1 - rho) * (ws * rs) @ powers
    diag = (1 - rho) ** 2 * rs**2 / np.maximum(rg, np.finfo(float).tiny)
    d2, d3 = np.sum(ws * diag * q**2), np.sum(ws * diag * q**3)
    x2, w2, segments = interval_grid(window["intervals"], n2)
    bnd = _boundary_data(jnp.asarray(x2), 1 / alpha, rho, Delta)
    _, kernel = _two_point_boundary_kernels(
        jnp.asarray(x2), 1 / alpha, rho, Delta, bnd, jnp.asarray(segments)
    )
    kernel = np.asarray(kernel)
    q2 = spectral_weight(x2, ell, q0, kappa)
    a, b = w2 * q2, w2 * q2**2
    return np.r_[
        m, s, d2 + a @ kernel @ a, d3 + 0.5 * (b @ kernel @ a + a @ kernel @ b)
    ]


def _weighted_contour(u, lo, hi, eta, ell, q0=0.05, kappa=0.0, *, strip_only=False):
    a, b = u**3, (1 - u) ** 3
    t = a / (a + b)
    dt = 3 * u * u * (1 - u) ** 2 / (a + b) ** 2
    angle = np.pi * t
    radius = (hi - lo) / 2
    height = min(radius, np.pi / (8 * ell)) if 0 < ell < np.inf else radius
    if 0 < ell < np.inf and eta + height >= np.pi / (4 * ell):
        raise ValueError("eta exceeds the finite-time pole-free contour strip.")
    x = (lo + hi) / 2 - radius * np.cos(angle)
    dx = radius * np.pi * dt * np.sin(angle)
    dhdx = np.zeros_like(x)
    if 0 < ell < np.inf and not strip_only:
        # A second, independent pole exclusion permits a taller contour at
        # large Re(z). The denominator is q0*z+((1-q0)*z-kappa)*exp(-2(z-kappa)ell).
        # For Re(z)>=kappa/[2(1-q0)], |(1-q0)-kappa/z|<=1-q0.
        # Thus its second term is strictly smaller than its first whenever
        # Re(z)>kappa+log((1-q0)/q0)/(2ell). No poles exist in that half-plane.
        bound = max(kappa / (2 * (1 - q0)), kappa + math.log((1 - q0) / q0) / (2 * ell))
        bound += 1e-10 * max(1.0, abs(bound))
        if lo > bound:
            height = radius
        elif hi > bound:
            v = np.clip((x - bound) / (hi - bound), 0, 1)
            dhdx = (radius - height) * 6 * v * (1 - v) / (hi - bound)
            height = height + (radius - height) * v * v * (3 - 2 * v)
    return (
        x + 1j * (eta + height * np.sin(angle)),
        dx + 1j * (dhdx * dx * np.sin(angle) + height * np.pi * dt * np.cos(angle)),
    )


def integrate_weighted_qmc(
    alpha,
    rho,
    Delta,
    *,
    window,
    ell,
    q0,
    kappa,
    eta,
    power=16,
    scrambles=8,
    seed=0,
    batch_size=4096,
    method="contour",
):
    """Full weighted signal chains through order three; no singular-term additions."""
    import jax.numpy as jnp
    from scipy.stats import qmc
    from scipy.interpolate import CubicSpline
    from . import chains as v7

    lo, hi = window["lambda_cut"], window["lambda_edge"]
    if not (eta > 0 and power >= 0 and scrambles >= 2 and batch_size > 0):
        raise ValueError("Invalid integration controls.")

    def path(u):
        if method == "real":
            return lo + (hi - lo) * u + 1j * eta, np.full_like(u, hi - lo)
        if method != "contour":
            raise ValueError("method must be contour or real.")
        return _weighted_contour(u, lo, hi, eta, ell, q0, kappa)

    started = time.perf_counter()
    grid = np.linspace(0, 1, 8193)
    zs, _ = path(grid)
    initial = np.asarray(v7._q_from_z(jnp.asarray(zs), 1 / alpha, rho, Delta))
    interp = CubicSpline(grid, -initial[:, 2])
    estimates, max_residual = [], 0.0
    size = min(batch_size, 2**power)
    for scramble in range(scrambles):
        sampler = qmc.Sobol(3, scramble=True, seed=seed + scramble)
        total = np.zeros(3)
        for start in range(0, 2**power, size):
            u = sampler.random(min(size, 2**power - start))
            zs, jac = path(u)
            q, residual, physical = v7._polish_q(
                jnp.asarray(zs), jnp.asarray(interp(u)), 1 / alpha, rho, Delta
            )
            max_residual = max(max_residual, float(residual))
            if (
                not bool(physical)
                or not np.isfinite(float(residual))
                or float(residual) > 1e-10
            ):
                raise ArithmeticError(
                    "Weighted integral: one-body branch failed residual check."
                )
            weighted_jac = jac * spectral_weight(zs, ell, q0, kappa)
            values = v7._discontinuity_weighted(
                q, jnp.asarray(weighted_jac), 1 / alpha, rho, Delta
            )
            total += np.asarray(values).sum(axis=0)
        estimates.append(total / 2**power)
    estimates = np.asarray(estimates)
    return dict(
        eta=eta,
        power=power,
        method=method,
        moments=estimates.mean(0),
        sem=estimates.std(0, ddof=1) / np.sqrt(scrambles),
        estimates=estimates,
        max_root_residual=max_residual,
        seconds=time.perf_counter() - started,
    )


def integrate_weighted_tensor(
    alpha, rho, Delta, *, window, ell, q0, kappa, eta, nodes=64, batch_size=4096
):
    """Independent deterministic contour GL integral, exploiting cubic symmetry.

    Only the sorted triples are evaluated, with exact permutation multiplicities.
    This resolves the near-diagonal structure on shallow finite-time contours
    much more efficiently than random points. No tensor is allocated in full.
    """
    import jax.numpy as jnp
    from scipy.special import roots_legendre
    from . import chains as v7

    started = time.perf_counter()
    # The pole-safe contour is piecewise analytic at x=bound. Split the GL
    # rule there rather than asking a global rule to resolve its C1 junction.
    cuts = [0.0, 1.0]
    if 0 < ell < np.inf:
        bound = max(kappa / (2 * (1 - q0)), kappa + math.log((1 - q0) / q0) / (2 * ell))
        bound += 1e-10 * max(1.0, abs(bound))
        lo, hi = window["lambda_cut"], window["lambda_edge"]
        if lo < bound < hi:
            t = np.arccos(((lo + hi) / 2 - bound) / ((hi - lo) / 2)) / np.pi
            ratio = np.cbrt(t / (1 - t))
            cuts.insert(1, float(ratio / (1 + ratio)))
    us = []
    ws = []
    for i, (left, right) in enumerate(zip(cuts, cuts[1:])):
        count = nodes // (len(cuts) - 1) + (1 if i < nodes % (len(cuts) - 1) else 0)
        x, w = roots_legendre(count)
        us.append(left + (right - left) * (x + 1) / 2)
        ws.append(w * (right - left) / 2)
    u, w = np.concatenate(us), np.concatenate(ws)
    z, jac = _weighted_contour(
        u, window["lambda_cut"], window["lambda_edge"], eta, ell, q0, kappa
    )
    q = v7._q_from_z(jnp.asarray(z), 1 / alpha, rho, Delta)
    jw = jac * w * spectral_weight(z, ell, q0, kappa)
    total = 0.0
    pending = []
    multiplicity = []

    def consume(indices, multiplicities):
        idx = np.asarray(indices)
        count = len(idx)
        if count < batch_size:
            idx = np.pad(idx, ((0, batch_size - count), (0, 0)), mode="edge")
        value = v7._discontinuity_weighted(
            q[idx], jnp.asarray(jw[idx]), 1 / alpha, rho, Delta
        )
        return float(np.asarray(value)[:count, -1] @ np.asarray(multiplicities))

    for i in range(nodes):
        for j in range(i, nodes):
            for k in range(j, nodes):
                pending.append((i, j, k))
                multiplicity.append(1 if i == k else 3 if i == j or j == k else 6)
                if len(pending) == batch_size:
                    total += consume(pending, multiplicity)
                    pending = []
                    multiplicity = []
    if pending:
        total += consume(pending, multiplicity)
    return dict(
        eta=eta,
        nodes=nodes,
        value=total,
        parameter_intervals=cuts,
        seconds=time.perf_counter() - started,
    )


def compute_weighted_traces_diagnostic(
    alpha,
    rho,
    Delta,
    *,
    phi=None,
    ell=np.inf,
    q0=0.05,
    kappa=0.0,
    integration_options=None,
):
    """Nine weighted constants and independent lower-grid/full-chain diagnostics.

    Options: n_grid1d, n_grid2, powers, eta_fractions, scrambles, seed,
    batch_size, rtol, atol, method, quadrature ('qmc' or 'tensor'), tensor_nodes.
    Failed convergence raises with diagnostics.
    """
    import jax
    from . import chains as v7

    _validate_flow(ell, q0, kappa)
    phi = validate_width(alpha, rho, Delta, phi)
    window = _active_window(find_window(alpha, rho, Delta, phi=phi), ell, kappa)
    opts = dict(
        n_grid1d=1024,
        n_grid2=256,
        powers=(14, 16, 18, 20),
        eta_fractions=(0.001, 0.0003, 0.0001, 0.00003),
        scrambles=8,
        seed=0,
        batch_size=4096,
        rtol=1e-3,
        atol=1e-5,
        method="contour",
        quadrature="qmc",
        tensor_nodes=(64, 128, 256, 512),
    )
    unknown = set(integration_options or {}) - set(opts)
    if unknown:
        raise ValueError(f"Unknown integration options: {unknown}")
    opts.update(integration_options or {})
    if opts["quadrature"] not in ("qmc", "tensor"):
        raise ValueError("quadrature must be qmc or tensor.")
    if opts["quadrature"] == "tensor" and (
        opts["method"] != "contour"
        or len(opts["tensor_nodes"]) < 3
        or any(a >= b for a, b in zip(opts["tensor_nodes"], opts["tensor_nodes"][1:]))
    ):
        raise ValueError(
            "Tensor quadrature requires contour and at least three increasing node counts."
        )
    if not (
        opts["rtol"] >= 0
        and opts["atol"] > 0
        and opts["n_grid1d"] >= 8
        and opts["n_grid2"] >= 8
        and len(opts["powers"]) >= 2
        and all(a < b for a, b in zip(opts["powers"], opts["powers"][1:]))
        and len(opts["eta_fractions"]) >= 3
        and all(e > 0 for e in opts["eta_fractions"])
        and all(a > b for a, b in zip(opts["eta_fractions"], opts["eta_fractions"][1:]))
    ):
        raise ValueError("Invalid refinement controls.")
    if not window["intervals"]:
        return dict(
            values=WeightedTraces(*([0.0] * 9)),
            absolute_error=np.zeros(9),
            converged=True,
            exact=True,
            window=window,
            phi=phi,
        )
    if phi == 1 and (ell == 0 or (np.isinf(ell) and kappa == 0)):
        # Only a constant, full-width weight makes the actual encoder Gram
        # scalar. Never apply this projector shortcut to general finite time.
        c = q0 if ell == 0 else 1.0
        s = 1 - rho
        return dict(
            values=WeightedTraces(
                c,
                c * c,
                c**3,
                s * c,
                s * c * c,
                s * c**3,
                s * c * c,
                s * c**3,
                s * c**3,
            ),
            absolute_error=np.zeros(9),
            converged=True,
            exact=True,
            window=window,
            phi=phi,
        )
    with jax.default_device(jax.devices("cpu")[0]):
        lower = _lower_traces(
            alpha, rho, Delta, window, ell, q0, kappa, opts["n_grid1d"], opts["n_grid2"]
        )
        fine = _lower_traces(
            alpha,
            rho,
            Delta,
            window,
            ell,
            q0,
            kappa,
            2 * opts["n_grid1d"],
            2 * opts["n_grid2"],
        )
        lower_error = np.abs(fine - lower)
        lower_ok = bool(
            np.all(lower_error <= np.maximum(opts["atol"], opts["rtol"] * np.abs(fine)))
        )
        if opts["quadrature"] == "tensor":
            history = []
            eta_results = []
            width = window["lambda_edge"] - window["lambda_cut"]
            eta_scale = min(width, np.pi / (8 * ell)) if 0 < ell < np.inf else width
            for fraction in opts["eta_fractions"]:
                previous = None
                for level, nodes in enumerate(opts["tensor_nodes"]):
                    row = integrate_weighted_tensor(
                        alpha,
                        rho,
                        Delta,
                        window=window,
                        ell=ell,
                        q0=q0,
                        kappa=kappa,
                        eta=fraction * eta_scale,
                        nodes=nodes,
                        batch_size=opts["batch_size"],
                    )
                    row["grid_error"] = (
                        abs(row["value"] - previous["value"]) if previous else None
                    )
                    tol = max(opts["atol"], opts["rtol"] * abs(row["value"]))
                    row["grid_converged"] = bool(
                        level >= 2 and row["grid_error"] <= tol
                    )
                    history.append(row)
                    previous = row
                    if row["grid_converged"]:
                        break
                eta_results.append(row)
            last = eta_results[-1]
            eta_error = float(
                np.max(np.abs(np.diff([r["value"] for r in eta_results[-3:]])))
            )
            tol = max(opts["atol"], opts["rtol"] * abs(last["value"]))
            ok = (
                lower_ok
                and all(r["grid_converged"] for r in eta_results[-3:])
                and eta_error <= tol
            )
            result = dict(
                values=WeightedTraces(*fine, last["value"]),
                absolute_error=np.r_[lower_error, max(eta_error, last["grid_error"])],
                converged=bool(ok),
                lower_converged=lower_ok,
                window=window,
                phi=phi,
                ell=ell,
                q0=q0,
                kappa=kappa,
                eta_error=np.array([0.0, 0.0, eta_error]),
                history=history,
                options=opts,
                cubic_method="tensor",
            )
            if not ok:
                raise v7.TraceConvergenceError(
                    "Weighted tensor traces did not converge.", result
                )
            return result
        history, eta_results = [], []
        width = window["lambda_edge"] - window["lambda_cut"]
        eta_scale = min(width, np.pi / (8 * ell)) if 0 < ell < np.inf else width
        for fraction in opts["eta_fractions"]:
            previous = None
            for power in opts["powers"]:
                row = integrate_weighted_qmc(
                    alpha,
                    rho,
                    Delta,
                    window=window,
                    ell=ell,
                    q0=q0,
                    kappa=kappa,
                    eta=fraction * eta_scale,
                    power=power,
                    **{
                        k: opts[k]
                        for k in ("scrambles", "seed", "batch_size", "method")
                    },
                )
                tol = np.maximum(opts["atol"], opts["rtol"] * np.abs(row["moments"]))
                row["sample_converged"] = bool(
                    previous is not None
                    and np.all(3 * row["sem"] <= tol)
                    and np.all(np.abs(row["moments"] - previous["moments"]) <= tol)
                )
                history.append(row)
                previous = row
                if row["sample_converged"]:
                    break
            eta_results.append(row)
    last = eta_results[-1]
    eta_error = np.max(
        np.abs(np.diff(np.stack([r["moments"] for r in eta_results[-3:]]), axis=0)),
        axis=0,
    )
    tol = np.maximum(opts["atol"], opts["rtol"] * np.abs(last["moments"]))
    chain_ok = all(r["sample_converged"] for r in eta_results[-3:]) and np.all(
        eta_error <= tol
    )
    error = np.r_[lower_error, max(eta_error[2], 3 * last["sem"][2])]
    result = dict(
        values=WeightedTraces(*fine, float(last["moments"][2])),
        absolute_error=error,
        converged=bool(lower_ok and chain_ok),
        lower_converged=lower_ok,
        window=window,
        phi=phi,
        ell=ell,
        q0=q0,
        kappa=kappa,
        eta_error=eta_error,
        history=history,
        options=opts,
        chain_lower_difference=last["moments"][:2] - fine[[3, 6]],
        cubic_method="qmc",
    )
    if not result["converged"]:
        raise v7.TraceConvergenceError("Weighted traces did not converge.", result)
    return result


def compute_weighted_traces(alpha, rho, Delta, **kwargs):
    """Host API returning WeightedTraces; see diagnostic API for controls."""
    return compute_weighted_traces_diagnostic(alpha, rho, Delta, **kwargs)["values"]
