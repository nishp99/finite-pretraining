"""Fixed-size physical operator surrogate with a cyclic weighted trace."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Geometry:
    bs: np.ndarray
    bn: np.ndarray
    weights: np.ndarray
    rho: float = 0.8
    delta: float = 0.1
    S: float = 1.0

    def __post_init__(self):
        self.bs = np.asarray(self.bs, float)
        self.bn = np.asarray(self.bn, float)
        self.weights = np.asarray(self.weights, float)
        n = len(self.weights)
        if (
            self.bs.shape != (n, n)
            or self.bn.shape != (n, n)
            or np.min(self.weights) <= 0
        ):
            raise ValueError("Invalid geometry dimensions/masses")
        if not (0 < self.rho < 1 and self.delta > 0 and self.S > 0):
            raise ValueError("Invalid population parameters")
        for b in (self.bs, self.bn):
            if (
                not np.allclose(b, b.T, atol=1e-12)
                or np.linalg.eigvalsh(b).min() < -1e-10
            ):
                raise ValueError("Non-PSD geometry")
            if (
                np.max(abs(self.weights[:, None] * b - b * self.weights[None, :]))
                > 1e-11
            ):
                raise ValueError("Trace weights must commute with each block")
        self.h = self.bs + self.bn
        self.c = self.bs + self.delta * self.bn
        self.ev, self.rotation = np.linalg.eigh(self.c)
        if self.ev.min() <= 0:
            raise ValueError(
                "Positive covariance required; remove exact null modes explicitly"
            )
        self.wteacher = self.S / (1 - self.rho) * self.bs
        self.teacher_power = float(
            np.trace(np.diag(self.weights) @ np.linalg.solve(self.c, self.wteacher))
        )
        if self.teacher_power > self.S + 1e-9:
            raise ValueError(
                f"Infeasible teacher power {self.teacher_power} > {self.S}"
            )
        u = self.rotation
        self.mass = np.diag(u.T @ np.diag(self.weights) @ u)
        if not np.allclose(
            u.T @ np.diag(self.weights) @ u, np.diag(self.mass), atol=1e-10
        ):
            # eigh may rotate exactly degenerate modes across differently weighted
            # blocks; diagonalize per block in callers instead of changing weights.
            raise ValueError("Covariance basis mixed unequal trace-weight blocks")
        mats = np.array([np.eye(n), u.T @ self.bs @ u, u.T @ self.bn @ u])
        w = u.T @ self.wteacher @ u
        self.jcoeff = self.mass[None, :] * np.diagonal(mats @ w, axis1=1, axis2=2)
        self.qcoeff = self.mass[None, :, None] * mats * w.T[None, :, :]
        self.ycoeff = self.mass[None, :] * np.diagonal(mats, axis1=1, axis2=2)
        self.teacher = self.jcoeff.sum(1)
        self.diffusion = self.ycoeff @ self.ev

    @property
    def dimension(self):
        return len(self.weights)

    @property
    def phi(self):
        return float(self.weights.sum())

    def packed(self):
        return self.ev, self.jcoeff, self.qcoeff, self.ycoeff, self.diffusion

    def moments(self):
        h = self.h
        b = self.bs
        tr = lambda x: np.sum(self.weights * np.diag(x))
        return np.array(
            [
                tr(h),
                tr(h @ h),
                tr(h @ h @ h),
                tr(b),
                tr(h @ b),
                tr(h @ h @ b),
                tr(b @ b),
                tr(h @ b @ b),
                tr(b @ b @ b),
            ]
        )

    def observables(self, f, y):
        return np.r_[
            self.jcoeff @ f,
            np.einsum("i,aij,j->a", f, self.qcoeff, f) + self.ycoeff @ y,
        ]

    def margins(self, state):
        j = state[:3]
        q = state[3:]
        var = q[1] + self.delta * q[2]
        return np.r_[
            q,
            self.teacher * q - j * j,
            self.teacher_power * var - j[0] ** 2,
            var - self.ev.min() * q[0],
            self.ev.max() * q[0] - var,
        ]


def from_ha(h, a, phi, rho=0.8, delta=0.1):
    h = np.asarray(h)
    root = np.sqrt(h)
    bs = root[:, None] * a * root[None, :]
    return Geometry(bs, np.diag(h) - bs, np.full(len(h), phi / len(h)), rho, delta)
