"""Rank-constrained C5 coordinates and trace contractions."""

import numpy as np
from scipy.fft import dct


def scales(target, errors=None):
    result = np.maximum(1e-8, 2e-4 * np.abs(target))
    if errors is not None:
        # Match incumbent: uncertainty estimates affect the six A contractions.
        result[..., 3:] = np.maximum(result[..., 3:], np.asarray(errors)[..., 3:])
    return result


def problem(target, hs, phi, rho):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    hs = np.atleast_2d(hs)
    m = hs.shape[1]
    w = phi / m
    rank = min(m, int(np.floor((1 - rho) / w + 1e-9)))
    ones = max(0, int(np.ceil((phi - rho) / w - 1e-9)))
    free = rank - ones
    if rank == 0 or free < 0:
        raise ValueError("No admissible rank allocation")
    basis = dct(np.eye(m), type=2, norm="ortho", axis=0).T
    mean = np.clip(
        np.mean(np.asarray(target)[..., 3] / np.asarray(target)[..., 0]) * m / rank,
        0.05,
        0.95,
    )
    center = np.log(mean / (1 - mean))
    initial = np.r_[basis[:, :rank].ravel(), center + 0.03 * np.linspace(-1, 1, free)]

    def unpack(x):
        q, _ = jnp.linalg.qr(x[: m * rank].reshape(m, rank), mode="reduced")
        logits = x[m * rank :]
        ev = jnp.concatenate((jnp.ones(ones), jax.nn.sigmoid(logits)))
        a = (q[:, :rank] * ev) @ q[:, :rank].T
        return a, logits

    @jax.jit
    def contractions(x):
        a, _ = unpack(x)

        def one(h):
            b = jnp.sqrt(h)[:, None] * a * jnp.sqrt(h)[None, :]
            bb = b @ b
            return w * jnp.array(
                [
                    jnp.sum(h),
                    jnp.sum(h * h),
                    jnp.sum(h**3),
                    jnp.trace(b),
                    jnp.sum(h * jnp.diag(b)),
                    jnp.sum(h * h * jnp.diag(b)),
                    jnp.trace(bb),
                    jnp.sum(h * jnp.diag(bb)),
                    jnp.trace(bb @ b),
                ]
            )

        return jax.vmap(one)(jnp.asarray(hs))

    @jax.jit
    def objective(x):
        a, logits = unpack(x)
        return jnp.sum(jax.nn.sigmoid(logits) * logits - jax.nn.softplus(logits)) / m

    return (
        initial,
        jax.jit(lambda x: unpack(x)[0]),
        contractions,
        objective,
        dict(
            m=m,
            rank=rank,
            unit_rank=ones,
            free_rank=free,
            initialization="ordered orthonormal DCT-II; free logit variation 0.03",
        ),
    )
