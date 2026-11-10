"""Boundary transforms and two-point density."""

import math
import jax.numpy as jnp
from jax import vmap


def _cubic_coeffs(z, gamma, rho, Delta):
    return jnp.array(
        [
            Delta,
            -(z * (1.0 + Delta) + Delta * (1.0 - gamma)),
            z * (z + 1.0 + Delta - gamma * ((1.0 - rho) + rho * Delta)),
            -(z * z),
        ],
        dtype=jnp.complex128,
    )


def _roots_cubic(coeffs):
    c = coeffs / coeffs[0]
    companion = jnp.array(
        [
            [0, 0, -c[3]],
            [1, 0, -c[2]],
            [0, 1, -c[1]],
        ],
        dtype=jnp.complex128,
    )
    return jnp.linalg.eigvals(companion)


def _choose_root_single(z, gamma, rho, Delta):
    coeffs = _cubic_coeffs(z, gamma, rho, Delta)
    roots = _roots_cubic(coeffs)
    sign = jnp.where(z.imag >= 0.0, 1.0, -1.0)
    g1 = 1.0 / (roots - z)
    g2 = 1.0 / (Delta * roots - z)
    m = (1.0 - rho) * g1 + rho * g2
    scores = sign * m.imag
    return roots[jnp.argmax(scores)]


def _boundary_data(xs, gamma, rho, Delta, eta_boundary=1e-10):
    # xs must be sorted ascending for branch tracking
    # Use vmap(_choose_root_single) so every node is evaluated independently.
    # This avoids the branch-tracking requirement that xs be densely and
    # uniformly spaced — essential when xs comes from a trig-substituted GL
    # grid whose spacing is non-uniform.
    zs_plus = xs.astype(jnp.complex128) + 1j * eta_boundary

    def _eval_upper(z):
        xi = _choose_root_single(z, gamma, rho, Delta)
        g1 = 1.0 / (xi - z)
        g2 = 1.0 / (Delta * xi - z)
        return xi, g1, g2

    xi_plus, g1_plus, g2_plus = vmap(_eval_upper)(zs_plus)

    xi_minus = jnp.conjugate(xi_plus)
    g1_minus = jnp.conjugate(g1_plus)
    g2_minus = jnp.conjugate(g2_plus)

    m_plus = (1.0 - rho) * g1_plus + rho * g2_plus

    return {
        "xi_plus": xi_plus,
        "xi_minus": xi_minus,
        "g1_plus": g1_plus,
        "g2_plus": g2_plus,
        "g1_minus": g1_minus,
        "g2_minus": g2_minus,
        "rho_global": m_plus.imag / math.pi,
        "rho_M": g1_plus.imag / math.pi,
    }


def _two_point_boundary_kernels(xs, gamma, rho, Delta, boundary, segments=None):
    xi_p = boundary["xi_plus"]
    xi_m = boundary["xi_minus"]
    g1_p = boundary["g1_plus"]
    g2_p = boundary["g2_plus"]
    g1_m = boundary["g1_minus"]
    g2_m = boundary["g2_minus"]
    rho_global = boundary["rho_global"]
    rho_M = boundary["rho_M"]

    diag_density = (1.0 - rho) ** 2 * (rho_M**2) / rho_global

    xi1 = xi_p[:, None]
    g11_p = g1_p[:, None]
    g21_p = g2_p[:, None]

    xi2_pp = xi_p[None, :]
    g12_pp = g1_p[None, :]
    g22_pp = g2_p[None, :]

    xi2_pm = xi_m[None, :]
    g12_pm = g1_m[None, :]
    g22_pm = g2_m[None, :]

    kappa_pp = (
        gamma
        * xi1
        * xi2_pp
        * ((1.0 - rho) * g11_p * g12_pp + rho * (Delta**2) * g21_p * g22_pp)
    )
    G_pp = (1.0 - rho) * (
        g11_p * g12_pp
        + gamma
        * (1.0 - rho)
        * xi1
        * xi2_pp
        * (g11_p**2)
        * (g12_pp**2)
        / (1.0 - kappa_pp)
    )

    kappa_pm = (
        gamma
        * xi1
        * xi2_pm
        * ((1.0 - rho) * g11_p * g12_pm + rho * (Delta**2) * g21_p * g22_pm)
    )
    G_pm = (1.0 - rho) * (
        g11_p * g12_pm
        + gamma
        * (1.0 - rho)
        * xi1
        * xi2_pm
        * (g11_p**2)
        * (g12_pm**2)
        / (1.0 - kappa_pm)
    )

    k_reg = -(jnp.real(G_pp) - jnp.real(G_pm)) / (2.0 * math.pi**2)

    if segments is None:
        segments = jnp.zeros(xs.shape[0], dtype=int)
    # Fill diagonal singularity from neighbors in the same support interval.
    n = xs.shape[0]
    idx = jnp.arange(n)

    def _fill_one(i):
        v1 = k_reg[i, i - 1]
        v2 = k_reg[i, i + 1]
        v3 = k_reg[i, i - 2]
        v4 = k_reg[i, i + 2]
        w1 = jnp.where((i > 0) & (segments[i] == segments[i - 1]), 1.0, 0.0)
        w2 = jnp.where((i + 1 < n) & (segments[i] == segments[i + 1]), 1.0, 0.0)
        w3 = jnp.where((i > 1) & (segments[i] == segments[i - 2]), 1.0, 0.0)
        w4 = jnp.where((i + 2 < n) & (segments[i] == segments[i + 2]), 1.0, 0.0)
        total_w = jnp.maximum(w1 + w2 + w3 + w4, 1.0)
        return (v1 * w1 + v2 * w2 + v3 * w3 + v4 * w4) / total_w

    diag_vals = vmap(_fill_one)(idx)
    k_reg = k_reg.at[idx, idx].set(diag_vals)

    return diag_density, k_reg
