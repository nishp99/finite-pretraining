"""Positive exponential midpoint integration of 2m annealed modal scalars.

The state dimension depends only on the chosen surrogate m, never ambient D.
All integration and reconstruction are explicitly on CPU.
"""

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp


def factors(state, ev, wdiag, mass, parameters, rl, eta, sphere, xp):
    delta, S, T, eps = parameters
    f, y = state
    j0 = xp.sum(mass * wdiag * f)
    q = xp.sum(mass * (wdiag * f * f + y))
    var = xp.sum(mass * ev * (wdiag * f * f + y))
    corr = j0 / xp.sqrt(var * S)
    # Roundoff-only arcsin protection; caller separately rejects physical exits.
    p = 0.5 + xp.arcsin(xp.clip(corr, -1.0, 1.0)) / xp.pi
    fac = eta * xp.where(rl, p ** (T - 1), 1.0) / xp.sqrt(2 * xp.pi)
    a = xp.where(rl, 1.0, -1.0) * fac / xp.sqrt(var)
    b = fac * xp.where(rl, 1.0, 1 - 2 * eps) / xp.sqrt(S)
    noise = eta**2 * xp.where(rl, p**T, 1 - (eps + (1 - 2 * eps) * p)) / T
    lam = xp.where(
        sphere, (2 * a * var + 2 * b * j0 + noise * xp.sum(mass * ev)) / (2 * q), 0.0
    )
    return a, b, noise, lam, corr, q


def objective_coefficients(state, ev, wdiag, mass, parameters, rl, xp=np):
    """Coefficients of Jdot - J Vdot/(2V), without the arcsin prefactor.

    Radial terms cancel, so the same coefficients apply on the sphere.
    The state must have positive variance and physical correlation.
    """
    a, b, h = factors(state, ev, wdiag, mass, parameters, rl, 1.0, False, xp)[:3]
    f, y = state
    u = wdiag * f * f + y
    j = xp.sum(mass * wdiag * f)
    v = xp.sum(mass * ev * u)
    jl = xp.sum(mass * ev * wdiag * f)
    v2 = xp.sum(mass * ev * ev * u)
    linear = a * jl + b * xp.sum(mass * wdiag) - j / v * (a * v2 + b * jl)
    quadratic = -j * h * xp.sum(mass * ev * ev) / (2 * v)
    return linear, quadratic


def optimal_rate(linear, quadratic, xp=np):
    """Exact bounded quadratic maximum; ties choose the smallest rate.

    Unlike the historical helper, permits zero and compares endpoints for
    convex objectives. No epsilon floor or arcsin derivative is needed.
    """
    safe = xp.where(quadratic < 0, quadratic, -1.0)
    vertex = xp.clip(-linear / (2 * safe), 0.0, 1.0)
    vertex = xp.where(quadratic < 0, vertex, 0.0)
    candidates = xp.stack((xp.zeros_like(vertex), vertex, xp.ones_like(vertex)))
    values = linear * candidates + quadratic * candidates * candidates
    rate = candidates[xp.argmax(values)]
    return xp.where(xp.isfinite(linear) & xp.isfinite(quadratic), rate, xp.nan)


def exprel(z):
    small = jnp.abs(z) < 1e-5
    safe = jnp.where(small, 1.0, z)
    return jnp.where(small, 1 + z / 2 + z * z / 6 + z**3 / 24, jnp.expm1(z) / safe)


@partial(jax.jit, static_argnames=("substeps", "record_every"))
def _run(
    ev,
    jcoef,
    qcoef,
    ycoef,
    mass,
    wdiag,
    params,
    rates,
    phases,
    sphere,
    dt,
    variance,
    optimal,
    initial,
    *,
    substeps,
    record_every,
):
    radius = variance * jnp.sum(mass)

    def advance(state, coeff, step):
        a, b, h, lam = coeff[:4]
        k = a * ev - lam
        out = jnp.stack(
            (
                jnp.exp(k * step) * state[0] + b * step * exprel(k * step),
                jnp.exp(2 * k * step) * state[1] + h * ev * step * exprel(2 * k * step),
            )
        )
        norm = jnp.sum(mass * (wdiag * out[0] ** 2 + out[1]))
        scale = radius / norm
        return jnp.where(
            sphere, jnp.stack((out[0] * jnp.sqrt(scale), out[1] * scale)), out
        )

    def step(state, args):
        index, prescribed, rl = args
        linear, quadratic = objective_coefficients(
            state, ev, wdiag, mass, params, rl, jnp
        )
        eta = jnp.where(optimal, optimal_rate(linear, quadratic, jnp), prescribed)
        start_valid = jnp.isfinite(linear) & jnp.isfinite(quadratic)

        def sub(_, s):
            coeff = factors(s, ev, wdiag, mass, params, rl, eta, sphere, jnp)
            half = advance(s, coeff, dt / (2 * substeps))
            mid = factors(half, ev, wdiag, mass, params, rl, eta, sphere, jnp)
            out = advance(s, mid, dt / substeps)
            corr = factors(out, ev, wdiag, mass, params, rl, eta, sphere, jnp)[4]
            valid = (
                jnp.all(jnp.isfinite(out))
                & jnp.all(out[1] >= 0)
                & (abs(corr) <= 1 + 1e-10)
            )
            return jnp.where(valid, out, jnp.full_like(out, jnp.nan))

        out = jax.lax.fori_loop(0, substeps, sub, state)
        f, y = out
        obs = jax.lax.cond(
            index % record_every == record_every - 1,
            lambda _: jnp.concatenate(
                (jcoef @ f, jnp.einsum("i,aij,j->a", f, qcoef, f) + ycoef @ y)
            ),
            lambda _: jnp.zeros(6),
            operand=None,
        )
        return out, (
            obs,
            jnp.all(jnp.isfinite(out)),
            jnp.where(start_valid, eta, jnp.nan),
            linear,
            quadratic,
        )

    return jax.lax.scan(step, initial, (jnp.arange(len(rates)), rates, phases))


def integrate(
    geometry,
    config,
    epochs=20,
    schedule="switch",
    switch=10,
    rates=None,
    substeps=1,
    student_variance=1.0,
    *,
    optimal=False,
    initial_modal_state=None,
    start_epoch=0,
):
    """Integrate on CPU; control is held for each microscopic interval 1/D.

    Diagnostic arrays have epochs*D entries evaluated before each interval.
    Explicit rates may exceed one; only adaptive rates are bounded by one.
    """
    if optimal and rates is not None:
        raise ValueError("optimal and explicit rates are mutually exclusive")
    if (
        substeps < 1
        or epochs < 1
        or not np.isfinite(student_variance)
        or student_variance <= 0
    ):
        raise ValueError("Invalid integration controls")
    if abs(config.delta - geometry.delta) > 1e-12 or abs(config.S - geometry.S) > 1e-12:
        raise ValueError("Geometry/configuration mismatch")
    if not isinstance(start_epoch, (int, np.integer)) or start_epoch < 0:
        raise ValueError("Invalid start epoch")
    n = config.D * epochs
    if schedule not in ("sft", "rl", "switch"):
        raise ValueError("Unknown schedule")
    phases = (
        np.ones(n, bool)
        if schedule == "rl"
        else (
            np.arange(n) + start_epoch * config.D >= switch * config.D
            if schedule == "switch"
            else np.zeros(n, bool)
        )
    )
    rates = np.full(n, config.eta) if rates is None else np.asarray(rates)
    if rates.shape != (n,) or not np.isfinite(rates).all() or np.min(rates) < 0:
        raise ValueError("Invalid rates")
    ev, jcoef, qcoef, ycoef, _ = geometry.packed()
    wdiag = np.diag(qcoef[0]) / geometry.mass
    if (
        not np.isfinite(student_variance * np.sum(geometry.mass * ev))
        or student_variance * np.sum(geometry.mass * ev) <= 0
    ):
        raise ValueError("Initial variance must be positive and finite")
    modal = (
        np.stack((np.zeros_like(ev), np.full_like(ev, student_variance)))
        if initial_modal_state is None
        else np.asarray(initial_modal_state, float)
    )
    if (
        modal.shape != (2, geometry.dimension)
        or not np.isfinite(modal).all()
        or np.any(modal[1] < 0)
    ):
        raise ValueError("Invalid modal checkpoint")
    initial = geometry.observables(*modal)
    if np.min(geometry.margins(initial)) < -1e-9:
        raise ValueError("Unphysical modal checkpoint")
    jax.config.update("jax_enable_x64", True)
    with jax.default_device(jax.devices("cpu")[0]):
        final_modal, (raw, valid, selected, linear, quadratic) = _run(
            ev,
            jcoef,
            qcoef,
            ycoef,
            geometry.mass,
            wdiag,
            np.array([config.delta, config.S, config.T, config.epsilon]),
            rates,
            phases,
            config.spherical,
            1 / config.D,
            student_variance,
            optimal,
            modal,
            substeps=substeps,
            record_every=config.D,
        )
        raw, valid = np.asarray(raw), np.asarray(valid)
    states = np.vstack((initial, raw[config.D - 1 :: config.D]))
    bad = np.flatnonzero(~valid)
    margins = np.array([geometry.margins(s) for s in states])
    if np.isfinite(margins).all() and margins.min() < -1e-9:
        raise ArithmeticError("Surrogate physical invariant violated")
    return dict(
        states=states,
        final_modal_state=np.asarray(final_modal),
        start_epoch=start_epoch,
        end_epoch=start_epoch + epochs,
        failed_step=int(bad[0]) if len(bad) else None,
        minimum_margin=float(np.nanmin(margins)),
        dynamic_dimension=2 * geometry.dimension,
        backend="cpu",
        rates=np.asarray(selected),
        objective_linear=np.asarray(linear),
        objective_quadratic=np.asarray(quadratic),
    )
