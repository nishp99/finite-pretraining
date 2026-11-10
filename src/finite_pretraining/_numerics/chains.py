"""Contiguous resolvent chains for the cubic trace."""

from itertools import combinations, product
import jax.numpy as jnp
from jax import jit, vmap, lax
from .boundary import _choose_root_single


def _pair_factors(qi, qj, gamma, rho, Delta):
    ss = qi[..., 0] * qj[..., 0]
    nn = qi[..., 1] * qj[..., 1]
    xx = gamma * qi[..., 2] * qj[..., 2]
    den = 1 - xx * ((1 - rho) * ss + rho * Delta**2 * nn)
    return ss, nn, xx, den


def _dress_factors(factors, source, rho, Delta):
    ss, nn, xx, den = factors
    u0 = (
        source[..., 2]
        + xx * ((1 - rho) * source[..., 0] + rho * Delta * source[..., 1])
    ) / den
    return jnp.stack(
        (source[..., 0] + ss * u0, source[..., 1] + Delta * nn * u0, u0), axis=-1
    )


def _chain_prefixes(q, gamma, rho, Delta):
    """All prefix chains in one pass, reusing contiguous subchains."""
    n = q.shape[-2]
    chains = {}
    for span in range(1, n):
        for i in range(n - span):
            j = i + span
            factors = _pair_factors(q[..., i, :], q[..., j, :], gamma, rho, Delta)
            if span == 1:
                ss = factors[0]
                source = jnp.stack((ss, jnp.zeros_like(ss), jnp.zeros_like(ss)), -1)
            else:
                source = jnp.zeros_like(q[..., 0, :])
                for count in range(1, span):
                    for cuts in combinations(range(i + 1, j), count):
                        points = (i,) + cuts + (j,)
                        term = chains[points[0], points[1]]
                        for a, b in zip(points[1:-1], points[2:]):
                            term = term * chains[a, b]
                        for cut in cuts:
                            term = term / q[..., cut, :]
                        source = source + (-1) ** (count + 1) * term
            chains[i, j] = _dress_factors(factors, source, rho, Delta)
    return jnp.stack(
        [(1 - rho) * q[..., 0, 0]]
        + [(1 - rho) * chains[0, j][..., 0] for j in range(1, n)],
        -1,
    )


@jit
def _q_from_z(z, gamma, rho, Delta):
    shape = z.shape
    xi = vmap(_choose_root_single, in_axes=(0, None, None, None))(
        z.reshape(-1), gamma, rho, Delta
    ).reshape(shape)
    return jnp.stack((1 / (xi - z), 1 / (Delta * xi - z), -xi), -1)


@jit
def _discontinuity_weighted(qplus, jacobian, gamma, rho, Delta):
    k = qplus.shape[-2]
    # Fix the first sign positive and add its conjugate partner analytically.
    signs = jnp.asarray([(1,) + s for s in product((1, -1), repeat=k - 1)])
    q = jnp.where(
        signs[None, ..., None] > 0, qplus[:, None, ...], jnp.conj(qplus[:, None, ...])
    )
    chains = _chain_prefixes(q, gamma, rho, Delta)
    signed_weights = jnp.where(
        signs[None, ...] > 0, jacobian[:, None, :], jnp.conj(jacobian[:, None, :])
    )
    chains = chains * jnp.cumprod(signed_weights, axis=-1)
    parity = jnp.cumprod(signs, axis=-1)
    powers = jnp.arange(1, k + 1)
    # Short prefixes repeat across unused signs; their mean removes duplication.
    return (
        jnp.mean(jnp.real(chains * parity / (2j * jnp.pi) ** powers), axis=1)
        * 2**powers
    )


@jit
def _polish_q(z, xi, gamma, rho, Delta):
    def step(_, xi):
        gs, gn = 1 / (xi - z), 1 / (Delta * xi - z)
        f = 1 / xi - 1 - gamma * ((1 - rho) * gs + rho * Delta * gn)
        df = -1 / xi**2 + gamma * ((1 - rho) * gs**2 + rho * Delta**2 * gn**2)
        return xi - f / df

    xi = lax.fori_loop(0, 4, step, xi)
    gs, gn = 1 / (xi - z), 1 / (Delta * xi - z)
    residual = jnp.max(
        jnp.abs(1 / xi - 1 - gamma * ((1 - rho) * gs + rho * Delta * gn))
        / (1 + jnp.abs(1 / xi))
    )
    physical = jnp.all((gs.imag > 0) & (gn.imag > 0))
    return jnp.stack((gs, gn, -xi), -1), residual, physical


class TraceConvergenceError(ArithmeticError):
    """Nonconverged result; diagnostic contains the completed run for inspection."""

    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic
