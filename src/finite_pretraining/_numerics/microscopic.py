"""Reusable batched microscopic updates and conditioned positive modal reference."""

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from .modal import exprel


def update_from_draws(
    v, ev, beta, remaining, xi, flips, D, T, epsilon, eta, rl, sphere, radius
):
    z = xi[:, :-1] * jnp.sqrt(ev)
    target = jnp.sign(xi[:, :-1] @ beta + remaining * xi[:, -1])
    pred = jnp.where(z @ v >= 0, 1.0, -1.0)
    noisy = target * jnp.where(flips < epsilon, -1.0, 1.0)
    coeff = jnp.where(rl, pred * jnp.all(pred == target), (noisy - pred) / 2)
    v = v + eta / (T * jnp.sqrt(float(D))) * (coeff @ z)
    return jnp.where(sphere, v * jnp.sqrt(radius / (v @ v)), v)


@partial(jax.jit, static_argnames=("D", "T"))
def microscopic_epoch(
    v, keys, ev, beta, remaining, radius, *, D, T, epsilon, eta, rl, sphere
):
    def single(v, key, ev, beta, remaining, radius):
        def step(index, carry):
            v, key = carry
            key, kn, kf = jax.random.split(key, 3)
            xi = jax.random.normal(kn, (T, v.shape[0] + 1), dtype=jnp.float64)
            flips = jax.random.uniform(kf, (T,))
            return update_from_draws(
                v,
                ev,
                beta,
                remaining,
                xi,
                flips,
                D,
                T,
                epsilon,
                jnp.broadcast_to(jnp.asarray(eta), (D,))[index],
                rl,
                sphere,
                radius,
            ), key

        return jax.lax.fori_loop(0, D, step, (v, key))

    return jax.vmap(single)(v, keys, ev, beta, remaining, radius)


def observables(v, t, mats, D):
    return (
        np.concatenate(
            (
                np.einsum("ni,aij,nj->na", v, mats, t),
                np.einsum("ni,aij,nj->na", v, mats, v),
            ),
            axis=-1,
        )
        / D
    )


@partial(
    jax.jit,
    static_argnames=(
        "D",
        "epochs",
        "switch",
        "substeps",
        "start_epoch",
        "return_checkpoint",
    ),
)
def conditioned_modes(
    v0,
    t,
    ev,
    mats,
    *,
    D,
    epochs,
    switch,
    T,
    epsilon,
    eta,
    sphere,
    schedule,
    substeps=1,
    initial_noise=None,
    start_epoch=0,
    return_checkpoint=False,
):
    """Evolve conditional mean and diagonal noise covariance; retain all six moments.

    The drift is the existing modal drift. Exponential midpoint integration
    preserves noise positivity and permits ordered off-diagonal reconstruction.
    schedule: 0=SFT, 1=RL, 2=switch. Batched over initial teacher/student pairs.
    """
    noise0 = jnp.zeros_like(v0) if initial_noise is None else initial_noise
    radius = jnp.sum(v0 * v0 + noise0, axis=-1)
    rates = jnp.broadcast_to(jnp.asarray(eta), (epochs * D,))

    def coeff(u, y, rl, rate):
        j = jnp.sum(u * t, axis=-1) / D
        q = jnp.sum(u * u + y, axis=-1) / D
        var = jnp.sum(ev * (u * u + y), axis=-1) / D
        p = 0.5 + jnp.arcsin(j / jnp.sqrt(var)) / jnp.pi
        fac = rate * jnp.where(rl, p ** (T - 1), 1.0) / jnp.sqrt(2 * jnp.pi)
        a = jnp.where(rl, 1.0, -1.0) * fac / jnp.sqrt(var)
        b = fac * jnp.where(rl, 1.0, 1 - 2 * epsilon)
        h = rate**2 * jnp.where(rl, p**T, 1 - (epsilon + (1 - 2 * epsilon) * p)) / T
        lam = jnp.where(
            sphere, (2 * a * var + 2 * b * j + h * jnp.sum(ev) / D) / (2 * q), 0.0
        )
        return a, b, h, lam

    def advance(u, y, c, dt):
        a, b, h, lam = (z[:, None] for z in c)
        k = a * ev - lam
        u = jnp.exp(k * dt) * u + b * t * dt * exprel(k * dt)
        y = jnp.exp(2 * k * dt) * y + h * ev * dt * exprel(2 * k * dt)
        scale = radius / jnp.sum(u * u + y, axis=-1)
        return jnp.where(sphere, u * jnp.sqrt(scale[:, None]), u), jnp.where(
            sphere, y * scale[:, None], y
        )

    def obs(state):
        u, y = state
        j = jnp.einsum("ni,aij,nj->na", u, mats, t) / D
        q = (
            jnp.einsum("ni,aij,nj->na", u, mats, u)
            + y @ jnp.diagonal(mats, axis1=1, axis2=2).T
        ) / D
        var = jnp.sum(ev * (u * u + y), axis=-1) / D
        valid = (
            jnp.all(jnp.isfinite(u), axis=-1)
            & jnp.all(jnp.isfinite(y), axis=-1)
            & jnp.all(y >= 0, axis=-1)
            & (j[:, 0] ** 2 <= var + 1e-9)
        )
        return jnp.where(valid[:, None], jnp.concatenate((j, q), axis=-1), jnp.nan)

    def epoch(state, ep):
        rl = (schedule == 1) | ((schedule == 2) & (ep + start_epoch >= switch))

        def step(index, state):
            rate = rates[ep * D + index // substeps]
            u, y = state
            c = coeff(u, y, rl, rate)
            half = advance(u, y, c, 0.5 / (D * substeps))
            return advance(u, y, coeff(*half, rl, rate), 1 / (D * substeps))

        state = jax.lax.fori_loop(0, D * substeps, step, state)
        return state, obs(state)

    initial = (v0, noise0)
    final, history = jax.lax.scan(epoch, initial, jnp.arange(epochs))
    result = jnp.concatenate((obs(initial)[None], history), axis=0)
    return (result, final) if return_checkpoint else result
