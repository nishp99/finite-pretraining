"""Experimental positive-time C5. Legacy fitting and defaults are unchanged.

Node calibration uses H moments only. Physical approximate predictions are
returned with explicit flags; they are not promoted to converged fits.
"""

import time
import numpy as np
from scipy.optimize import least_squares, minimize
from .parameterization import scales, problem
from .geometry import from_ha

OBJECTIVES = ("entropy", "diag", "full") + tuple(
    f"entropy_{kind}_{gamma:g}"
    for kind in ("diag", "full")
    for gamma in (0.01, 0.1, 1.0, 10.0)
)


def positive_times(times):
    times = np.asarray(times, float)
    if (
        times.ndim != 1
        or not len(times)
        or np.any(np.isnan(times))
        or np.any(times <= 0)
    ):
        raise ValueError("C5 positive-time fits require ell > 0 (infinity is allowed)")
    return times


def weights_and_derivative(nodes, times, q0=0.05):
    from scipy.special import expit

    times = positive_times(times)
    if not 0 < q0 < 1:
        raise ValueError("Require 0 < q0 < 1")
    h = np.ones((len(times), len(nodes)))
    derivative = np.zeros_like(h)
    finite = np.isfinite(times)
    t = times[finite, None]
    h[finite] = expit(np.log(q0 / (1 - q0)) + 2 * t * np.asarray(nodes)[None, :])
    derivative[finite] = 2 * t * h[finite] * (1 - h[finite])
    return h, derivative


def calibrate_nodes(
    traces, times, phi, lambda_samples, bands, m=12, errors=None, q0=0.05
):
    times = positive_times(times)
    target = np.asarray(traces, float)
    anchor = np.quantile(lambda_samples, (np.arange(m) + 0.5) / m)
    bounds = np.array(
        [
            next(
                (b for b in bands if b[0] - 1e-10 <= x <= b[1] + 1e-10),
                (np.nan, np.nan),
            )
            for x in anchor
        ]
    )
    if not np.isfinite(bounds).all() or np.any(anchor <= 0):
        return dict(
            valid_nodes=False, feasible=False, reason="No positive in-band anchor"
        )
    scale = scales(target, errors)[:, :3]

    def residual(x):
        h, _ = weights_and_derivative(x, times, q0)
        return (
            (
                phi * np.stack([np.mean(h**p, axis=1) for p in (1, 2, 3)], axis=1)
                - target[:, :3]
            )
            / scale
        ).ravel()

    def jac(x):
        h, dh = weights_and_derivative(x, times, q0)
        return np.stack(
            [
                phi / m * p * h ** (p - 1) * dh / scale[:, i, None]
                for i, p in enumerate((1, 2, 3))
            ],
            axis=1,
        ).reshape(-1, m)

    first = least_squares(
        residual,
        anchor,
        jac=jac,
        bounds=(bounds[:, 0], bounds[:, 1]),
        max_nfev=1500,
        gtol=1e-10,
        ftol=1e-12,
        xtol=1e-12,
    )
    candidates = [("least_squares", np.sort(first.x))]

    # Epigraph minimax with explicit ordering. Sorting is not used inside derivatives.
    def constraint(z):
        return np.r_[
            z[-1] - residual(z[:-1]), z[-1] + residual(z[:-1]), np.diff(z[:-1])
        ]

    order = np.diff(np.eye(m), axis=0)

    def cj(z):
        j = jac(z[:-1])
        return np.vstack(
            (
                np.c_[-j, np.ones(len(j))],
                np.c_[j, np.ones(len(j))],
                np.c_[order, np.zeros(m - 1)],
            )
        )

    x = candidates[0][1]
    z = np.r_[x, np.max(abs(residual(x)))]
    gradient = np.r_[np.zeros(m), 1.0]
    second = minimize(
        lambda z: z[-1],
        z,
        jac=lambda z: gradient,
        method="SLSQP",
        bounds=[tuple(b) for b in bounds] + [(0, None)],
        constraints=[dict(type="ineq", fun=constraint, jac=cj)],
        callback=lambda z: candidates.append(("minimax_checkpoint", z[:-1].copy())),
        options=dict(maxiter=2000, ftol=1e-10),
    )
    candidates.append(("minimax_terminal", second.x[:-1]))
    valid = [
        (name, x)
        for name, x in candidates
        if np.isfinite(x).all()
        and np.all(x >= bounds[:, 0])
        and np.all(x <= bounds[:, 1])
        and np.all(np.diff(x) >= 0)
    ]
    name, nodes = min(
        valid,
        key=lambda c: (np.max(abs(residual(c[1]))), np.linalg.norm(residual(c[1]))),
    )
    r = residual(nodes)
    return dict(
        valid_nodes=True,
        feasible=bool(np.max(abs(r)) <= 1),
        nodes=nodes,
        anchor=anchor,
        bounds=bounds,
        normalized_residual=r,
        maximum_tolerance_ratio=float(np.max(abs(r))),
        residual_l2=float(np.linalg.norm(r)),
        candidate_origin=name,
        stage1_nodes=first.x,
        terminal_nodes=second.x[:-1],
        stage1_nfev=first.nfev,
        stage1_message=str(first.message),
        stage2_success=bool(second.success),
        stage2_message=str(second.message),
        stage2_nit=second.nit,
    )


def objective_function(unpack, entropy, method, profile, m):
    import jax
    import jax.numpy as jnp

    if method not in OBJECTIVES:
        raise ValueError("Unknown objective " + method)
    if method != "entropy":
        profile = np.asarray(profile, float)
        if (
            profile.shape != (m,)
            or not np.isfinite(profile).all()
            or np.any(profile < 0)
            or np.any(profile > 1)
        ):
            raise ValueError("Conditional profile must be finite and in [0,1]")

    @jax.jit
    def objective(x):
        if method == "entropy":
            return entropy(x)
        kind = method if method in ("diag", "full") else method.split("_")[1]
        a = unpack(x)
        p = jnp.asarray(profile)
        penalty = (
            jnp.sum((jnp.diag(a) - p) ** 2) / m
            if kind == "diag"
            else jnp.sum((a - jnp.diag(p)) ** 2) / m
        )
        return (
            penalty
            if method in ("diag", "full")
            else entropy(x) + float(method.split("_")[2]) * penalty
        )

    return objective


def select_candidate(candidates, terminal_success):
    """No downstream data or restarts enter this prescribed ordering."""
    terminal = candidates[-1]
    if terminal_success and terminal["physical_valid"] and terminal["trace_feasible"]:
        return terminal
    feasible = [c for c in candidates if c["physical_valid"] and c["trace_feasible"]]
    if feasible:
        return min(feasible, key=lambda c: (c["objective"], c["iteration"]))
    physical = [c for c in candidates if c["physical_valid"]]
    return (
        min(
            physical,
            key=lambda c: (
                c["maximum_tolerance_ratio"],
                c["residual_l2"],
                c["iteration"],
            ),
        )
        if physical
        else None
    )


def initialization_parameters(
    current, report, profile, initialization="current", rotation_amplitude=0.05
):
    """Prescribed initialization only; profile targets and fit constraints are unchanged."""
    if initialization == "current":
        return current.copy()
    if initialization != "profile":
        raise ValueError("Unknown initialization: " + initialization)
    m = report["m"]
    p = np.asarray(profile, float)
    if p.shape != (m,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Conditional profile must be finite and in [0,1]")
    if report["rank"] != m or report["unit_rank"] != 0:
        raise ValueError("Profile initialization requires full free signal rank")
    if not np.isfinite(rotation_amplitude) or rotation_amplitude < 0:
        raise ValueError("Invalid rotation amplitude")
    from scipy.linalg import expm
    from scipy.special import logit

    i, j = np.indices((m, m))
    distance = j - i
    g = np.divide(1.0, distance, out=np.zeros((m, m)), where=distance != 0)
    g /= np.linalg.norm(g, 2)
    q = expm(rotation_amplitude * g)
    # Only exact endpoints are moved, and only in the finite-logit initial state.
    finite = np.where(p == 0, 1e-8, np.where(p == 1, 1 - 1e-8, p))
    return np.r_[q.ravel(), logit(finite)]


def solve_geometry(
    traces,
    hs,
    phi,
    rho,
    delta,
    method="entropy",
    errors=None,
    profile=None,
    *,
    diagnostic_seed=None,
    initialization="current",
    rotation_amplitude=0.05,
):
    import jax
    import jax.numpy as jnp

    started = time.monotonic()
    target = np.atleast_2d(traces)
    hs = np.atleast_2d(hs)
    scale = scales(target, errors)
    x, unpack, values, entropy, report = problem(target, hs, phi, rho)
    x = initialization_parameters(
        x, report, profile, initialization, rotation_amplitude
    )
    initial_parameters = x.copy()
    if diagnostic_seed is not None:
        x = x + np.random.default_rng(diagnostic_seed).normal(0, 0.3, len(x))
    objective = objective_function(unpack, entropy, method, profile, hs.shape[1])
    residual = jax.jit(lambda z: (values(z) - jnp.asarray(target)) / jnp.asarray(scale))
    # H is fixed: impossible node moments must not block A's constrained solve.
    ar = jax.jit(lambda z: residual(z)[:, 3:].ravel())
    aj = jax.jit(jax.jacfwd(ar))
    grad = jax.jit(jax.grad(objective))
    f = lambda z: np.asarray(ar(z))
    jac = lambda z: np.asarray(aj(z))
    candidates = []

    def capture(z, origin):
        r = np.asarray(residual(z))
        a = np.asarray(unpack(z))
        physical = True
        error = None
        if not np.isfinite(r).all() or not np.isfinite(a).all():
            candidates.append(
                dict(
                    iteration=len(candidates),
                    origin=origin,
                    parameters=z.copy(),
                    a=a,
                    objective=float("inf"),
                    normalized_residual=r,
                    maximum_tolerance_ratio=float("inf"),
                    residual_l2=float("inf"),
                    residual_rms=float("inf"),
                    trace_feasible=False,
                    physical_valid=False,
                    physical_error="Nonfinite optimizer iterate",
                )
            )
            return
        try:
            gs = [from_ha(h, a, phi, rho, delta) for h in hs]
        except (ValueError, np.linalg.LinAlgError) as exc:
            physical = False
            error = str(exc)
            gs = []
        ae = np.linalg.eigvalsh(a)
        c = dict(
            iteration=len(candidates),
            origin=origin,
            parameters=z.copy(),
            a=a,
            objective=float(objective(z)),
            normalized_residual=r,
            maximum_tolerance_ratio=float(np.max(abs(r))),
            residual_l2=float(np.linalg.norm(r)),
            residual_rms=float(np.sqrt(np.mean(r * r))),
            trace_feasible=bool(np.max(abs(r)) <= 1),
            physical_valid=physical,
            physical_error=error,
            physical_margins=dict(
                a_min=float(ae.min()),
                one_minus_a_min=float(1 - ae.max()),
                covariance_min=min((g.ev.min() for g in gs), default=None),
                teacher_power=min((g.S - g.teacher_power for g in gs), default=None),
                signal_rank_mass=(1 - rho) - report["rank"] * phi / report["m"],
                noise_rank_mass=rho
                - (report["m"] - report["unit_rank"]) * phi / report["m"],
            ),
        )
        candidates.append(c)

    first = least_squares(
        f, x, jac=jac, max_nfev=1500, gtol=1e-8, ftol=1e-10, xtol=1e-10
    )
    capture(first.x, "stage1")
    try:
        second = minimize(
            lambda z: float(objective(z)),
            first.x,
            jac=lambda z: np.asarray(grad(z)),
            method="SLSQP",
            constraints=[
                dict(
                    type="ineq",
                    fun=lambda z: np.r_[1 - 1e-7 - f(z), 1 - 1e-7 + f(z)],
                    jac=lambda z: np.vstack((-jac(z), jac(z))),
                )
            ],
            callback=lambda z: capture(z, "stage2_checkpoint"),
            options=dict(maxiter=2000, ftol=1e-10),
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        from types import SimpleNamespace

        second = SimpleNamespace(
            x=candidates[-1]["parameters"],
            success=False,
            message=str(exc),
            nit=len(candidates) - 1,
        )
    capture(second.x, "terminal")
    chosen = select_candidate(candidates, bool(second.success))
    # Retain full states for stage1/terminal/best feasible, compact metadata for other iterates.
    feasible = [c for c in candidates if c["physical_valid"] and c["trace_feasible"]]
    best = (
        min(feasible, key=lambda c: (c["objective"], c["iteration"]))
        if feasible
        else None
    )
    report.update(
        method=method,
        nominal=diagnostic_seed is None,
        diagnostic_seed=diagnostic_seed,
        initialization=initialization,
        rotation_amplitude=rotation_amplitude,
        initial_parameters=initial_parameters,
        solver="SLSQP",
        stage1_nfev=first.nfev,
        stage1_message=str(first.message),
        stage2_success=bool(second.success),
        stage2_message=str(second.message),
        stage2_nit=second.nit,
        stage1=candidates[0],
        terminal=candidates[-1],
        best_feasible=best,
        history=[
            {
                k: v
                for k, v in c.items()
                if k not in ("parameters", "a", "normalized_residual")
            }
            for c in candidates
        ],
        h=hs,
        scale=scale,
        elapsed_seconds=time.monotonic() - started,
    )
    if chosen is None:
        return None, dict(report, status="no_physical_geometry", physical_valid=False)
    report.update(chosen)
    report["candidate_origin"] = chosen["origin"]
    report["solver_success"] = bool(second.success)
    report["status"] = (
        "fit_converged"
        if chosen is candidates[-1] and second.success and chosen["trace_feasible"]
        else "physical_approximate"
    )
    return [from_ha(h, chosen["a"], phi, rho, delta) for h in hs], report
