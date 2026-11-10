"""Host-side positive spectral support and upper-phi quantiles.

rho is the noise fraction. The companion Silverstein map gives support edges;
no density scan through the sample null atom is used. All masses exclude zero.
"""

from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_legendre


def validate_width(alpha, rho, Delta, phi=None):
    alpha, rho, Delta = map(float, (alpha, rho, Delta))
    phi = 1 - rho if phi is None else float(phi)
    if not (
        np.all(np.isfinite([alpha, rho, Delta, phi]))
        and alpha > 0
        and 0 < rho < 1
        and 0 < Delta < 1
        and 0 < phi <= 1
    ):
        raise ValueError("Require alpha>0, 0<rho<1, 0<Delta<1 and 0<phi<=1.")
    if alpha < 1 and phi >= alpha:
        raise ValueError(
            "Require phi<alpha when alpha<1 (no sample-nullspace selection)."
        )
    return phi


@lru_cache(maxsize=32)
def legendre_rule(n):
    return roots_legendre(n)


def interval_grid(intervals, n):
    """Endpoint-graded GL nodes for each occupied support interval."""
    u, w = legendre_rule(n)
    theta, wt = (u + 1) * np.pi / 4, w * np.pi / 4
    xs, ws, labels = [], [], []
    for i, (lo, hi) in enumerate(intervals):
        xs.append(lo + (hi - lo) * np.sin(theta) ** 2)
        ws.append(wt * (hi - lo) * np.sin(2 * theta))
        labels.append(np.full(n, i))
    return np.concatenate(xs), np.concatenate(ws), np.concatenate(labels)


def densities(xs, alpha, rho, Delta, eta=1e-12):
    z = np.asarray(xs, dtype=np.complex128) + 1j * eta
    gamma = 1 / alpha
    c1 = -(z * (1 + Delta) + Delta * (1 - gamma)) / Delta
    c2 = z * (z + 1 + Delta - gamma * ((1 - rho) + rho * Delta)) / Delta
    c3 = -z * z / Delta
    companion = np.zeros(z.shape + (3, 3), dtype=np.complex128)
    companion[..., 1, 0] = companion[..., 2, 1] = 1
    companion[..., 0, 2] = -c3
    companion[..., 1, 2] = -c2
    companion[..., 2, 2] = -c1
    xi = np.linalg.eigvals(companion)
    gs = 1 / (xi - z[..., None])
    gn = 1 / (Delta * xi - z[..., None])
    m = (1 - rho) * gs + rho * gn
    idx = np.argmax(m.imag, axis=-1)[..., None]
    return (
        np.maximum(0, np.take_along_axis(m.imag, idx, -1)[..., 0] / np.pi),
        np.maximum(0, np.take_along_axis(gs.imag, idx, -1)[..., 0] / np.pi),
    )


@lru_cache(maxsize=256)
def _direct_support(alpha, rho, Delta):
    # z(v)=-1/v+gamma*[s/(1+v)+rho*Delta/(1+Delta*v)].
    from numpy.polynomial import Polynomial as P

    v, a, b = P([0, 1]), P([1, 1]), P([1, Delta])
    gamma = 1 / alpha
    poly = a * a * b * b - gamma * v * v * ((1 - rho) * b * b + rho * Delta**2 * a * a)
    coeff = poly.coef.copy()
    while len(coeff) > 1 and abs(coeff[-1]) < 1e-14 * np.max(np.abs(coeff)):
        coeff = coeff[:-1]
    roots = P(coeff).roots()
    real = roots.real[np.abs(roots.imag) < 1e-7]
    edges = [
        -1 / q + gamma * ((1 - rho) / (1 + q) + rho * Delta / (1 + Delta * q))
        for q in real
    ]
    edges = sorted(float(x) for x in edges if x > 1e-13)
    if abs(alpha - 1) < 1e-13:
        edges.insert(0, 0.0)
    if len(edges) not in (2, 4):
        raise ArithmeticError(f"Could not resolve positive support: {edges}")
    return tuple(zip(edges[::2], edges[1::2]))


@lru_cache(maxsize=256)
def _window(alpha, rho, Delta, phi):
    intervals = support_intervals(alpha, rho, Delta)

    def mass(lo, hi, n=512):
        if hi <= lo:
            return 0.0
        x, w, _ = interval_grid(((lo, hi),), n)
        return float(w @ densities(x, alpha, rho, Delta)[0])

    masses = [mass(lo, hi) for lo, hi in intervals]
    selected, remaining, cut = [], phi, intervals[0][0]
    mass_tol = 2e-7
    for (lo, hi), m in reversed(list(zip(intervals, masses))):
        if remaining >= m - mass_tol:
            selected.insert(0, (lo, hi))
            remaining -= m
            cut = lo
            if remaining <= mass_tol:
                break
        else:
            cut = brentq(lambda c: mass(c, hi) - remaining, lo, hi, xtol=1e-13)
            selected.insert(0, (cut, hi))
            remaining = 0.0
            break
    selected_mass = sum(mass(lo, hi, 1024) for lo, hi in selected)
    if abs(selected_mass - phi) > 2e-6:
        raise ArithmeticError(f"Selected mass {selected_mass} differs from phi={phi}.")
    gap = len(intervals) == 2 and abs(cut - intervals[-1][0]) < 1e-10
    return dict(
        lambda_min=intervals[0][0],
        lambda_max=intervals[-1][1],
        lambda_edge=intervals[-1][1],
        lambda_cut=cut,
        lambda_cut_mass=cut,
        has_gap=gap,
        mass_continuous=min(alpha, 1.0),
        selected_mass=selected_mass,
        mass_error=selected_mass - phi,
        phi=phi,
        intervals=tuple(selected),
        support_intervals=intervals,
    )


def _direct_window(alpha, rho, Delta, scan_points=2000, *, phi=None):
    """Return the upper-phi window; scan_points is retained for compatibility."""
    phi = validate_width(alpha, rho, Delta, phi)
    return dict(_window(float(alpha), float(rho), float(Delta), phi))


def support_intervals(alpha, rho, delta):
    try:
        return _direct_support(alpha, rho, delta)
    except ArithmeticError as original:
        from numpy.polynomial import Polynomial as P

        w = P([0, 1])
        a = w + 1
        b = w + delta
        gamma = 1 / alpha
        polynomial = a * a * b * b - gamma * (
            (1 - rho) * b * b + rho * delta**2 * a * a
        )
        roots = polynomial.roots()
        real = roots.real[abs(roots.imag) < 1e-9]
        edges = []
        residuals = []
        for q in real:
            residual = 1 - gamma * (
                (1 - rho) / (q + 1) ** 2 + rho * delta**2 / (q + delta) ** 2
            )
            z = q * (-1 + gamma * ((1 - rho) / (q + 1) + rho * delta / (q + delta)))
            if z > 0:
                edges.append(float(z))
                residuals.append(float(residual))
        if abs(alpha - 1) < 1e-13:
            edges = [x for x in edges if x > 1e-13] + [0.0]
        edges.sort()
        if len(edges) not in (2, 4) or max(map(abs, residuals), default=0) > 1e-7:
            raise original
        result = tuple(zip(edges[::2], edges[1::2]))
        return result


def find_window(alpha, rho, delta, scan_points=2000, *, phi=None):
    phi = validate_width(alpha, rho, delta, phi)
    try:
        return _direct_window(alpha, rho, delta, scan_points, phi=phi)
    except ArithmeticError as original:
        intervals = support_intervals(alpha, rho, delta)
        for n in (1024, 2048, 4096):

            def mass(lo, hi, order=n):
                if hi <= lo:
                    return 0.0
                x, w, _ = interval_grid(((lo, hi),), order)
                return float(w @ densities(x, alpha, rho, delta)[0])

            remaining = phi
            selected = []
            cut = intervals[0][0]
            for lo, hi in reversed(intervals):
                val = mass(lo, hi)
                if remaining >= val - 2e-7:
                    selected.insert(0, (lo, hi))
                    remaining -= val
                    cut = lo
                    if remaining <= 2e-7:
                        break
                else:
                    cut = brentq(lambda x: mass(x, hi) - remaining, lo, hi, xtol=1e-13)
                    selected.insert(0, (cut, hi))
                    remaining = 0.0
                    break
            verified = sum(mass(lo, hi, 2 * n) for lo, hi in selected)
            # Independent adaptive quadrature check, not the calibration GL rule.
            from scipy.integrate import quad

            independent = sum(
                quad(
                    lambda theta: (
                        float(
                            densities(
                                np.array([lo + (hi - lo) * np.sin(theta) ** 2]),
                                alpha,
                                rho,
                                delta,
                            )[0][0]
                        )
                        * (hi - lo)
                        * np.sin(2 * theta)
                    ),
                    0,
                    np.pi / 2,
                    epsabs=1e-8,
                    epsrel=1e-8,
                    limit=500,
                )[0]
                for lo, hi in selected
            )
            if abs(verified - phi) <= 2e-6 and abs(independent - phi) <= 2e-6:
                return dict(
                    quadrature_refinement=n,
                    independent_mass=independent,
                    lambda_min=intervals[0][0],
                    lambda_max=intervals[-1][1],
                    lambda_edge=intervals[-1][1],
                    lambda_cut=cut,
                    lambda_cut_mass=cut,
                    has_gap=len(intervals) == 2 and abs(cut - intervals[-1][0]) < 1e-10,
                    mass_continuous=min(alpha, 1.0),
                    selected_mass=verified,
                    mass_error=verified - phi,
                    phi=phi,
                    intervals=tuple(selected),
                    support_intervals=intervals,
                )
        raise original
