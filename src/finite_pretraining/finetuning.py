"""CPU modal predictions and references; CUDA microscopic fine-tuning."""

from dataclasses import dataclass, asdict
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import eigh
from ._numerics.geometry import Geometry
from ._numerics.modal import integrate
from ._numerics.microscopic import conditioned_modes, microscopic_epoch, observables
from .pretraining import Encoder

OBSERVABLES = ("J0", "Js", "Jn", "Q0", "Qs", "Qn")


@dataclass(frozen=True)
class Dynamics:
    D: int = 800
    T: int = 7
    epsilon: float = 0.05
    eta: float = 1.0
    spherical: bool = True
    schedule: str = "switch"
    switch: int = 10
    epochs: int = 20
    policy: str = "adaptive"

    def __post_init__(self):
        if (
            any(not isinstance(i, int) or i < 1 for i in (self.D, self.T, self.epochs))
            or not isinstance(self.switch, int)
            or self.switch < 0
        ):
            raise ValueError("Invalid D, T, epochs or switch")
        if self.schedule not in ("sft", "rl", "switch") or self.policy not in (
            "adaptive",
            "fixed",
            "replay",
        ):
            raise ValueError("Unknown learning scenario or rate policy")
        if not (0 <= self.epsilon <= 1 and np.isfinite(self.eta) and self.eta >= 0):
            raise ValueError("Invalid noise or learning rate")


@dataclass
class Trajectory:
    epoch: np.ndarray
    states: np.ndarray
    accuracy: np.ndarray
    rates: np.ndarray
    status: str
    checkpoint: dict
    backend: str

    @property
    def observables(self):
        return {name: self.states[..., i] for i, name in enumerate(OBSERVABLES)}


def _result(states, rates, delta, status, checkpoint, backend, start):
    states = np.asarray(states)
    v = states[..., 4] + delta * states[..., 5]
    if not np.isfinite(states).all() or np.any(v <= 0):
        raise ArithmeticError("Invalid trajectory variance or state")
    corr = states[..., 0] / np.sqrt(v)
    if np.any(abs(corr) > 1 + 1e-12):
        raise ArithmeticError("Unphysical teacher correlation")
    return Trajectory(
        np.arange(start, start + len(states)),
        states,
        0.5 + np.arcsin(np.clip(corr, -1, 1)) / np.pi,
        np.asarray(rates),
        status,
        checkpoint,
        backend,
    )


def _rates(d, rates):
    if rates is None:
        if d.policy == "fixed":
            return np.full(d.D * d.epochs, d.eta)
        raise ValueError(
            "Supply the per-update schedule from predict(...).rates for matched replay"
        )
    rates = np.asarray(rates, float)
    if (
        rates.shape != (d.D * d.epochs,)
        or not np.isfinite(rates).all()
        or rates.min() < 0
    ):
        raise ValueError("Expected epochs*D nonnegative finite rates")
    return rates


def _modal(g, d, rates, checkpoint, status):
    from types import SimpleNamespace

    c = SimpleNamespace(**asdict(d), delta=g.delta, S=g.S)
    start = checkpoint["epoch"] if checkpoint else 0
    result = integrate(
        g,
        c,
        epochs=d.epochs,
        schedule=d.schedule,
        switch=d.switch,
        rates=rates,
        optimal=d.policy == "adaptive" and rates is None,
        initial_modal_state=checkpoint["state"] if checkpoint else None,
        start_epoch=start,
    )
    if result["failed_step"] is not None:
        raise ArithmeticError(
            f"ODE left its physical domain at update {result['failed_step']}"
        )
    return _result(
        result["states"],
        result["rates"],
        g.delta,
        status,
        dict(epoch=result["end_epoch"], state=result["final_modal_state"]),
        "cpu",
        start,
    )


def predict(
    fit,
    pretraining_time,
    *,
    dynamics=None,
    rates=None,
    checkpoint=None,
    allow_approximate=False,
):
    """Evolve C5 on CPU. Accuracy is a fraction. A checkpoint resumes at its absolute epoch."""
    d = dynamics or Dynamics()
    if rates is not None and d.policy != "replay":
        raise ValueError('Use policy="replay" with supplied rates')
    return _modal(
        fit.geometry(pretraining_time, allow_approximate=allow_approximate),
        d,
        _rates(d, rates) if d.policy != "adaptive" else None,
        checkpoint,
        fit.status,
    )


def _actual(encoder, d, pairs, seed):
    e = encoder.weights
    k, D = e.shape
    s = encoder.signal_dim
    if D != d.D or pairs < 1:
        raise ValueError("Encoder D and dynamics D must match; pairs must be positive")
    bs = e[:, :s] @ e[:, :s].T
    bn = e[:, s:] @ e[:, s:].T
    ev, u = eigh(bs + encoder.delta * bn, driver="evd")
    if ev.min() <= 0:
        raise ValueError(
            "Actual downstream encoder requires positive feature covariance"
        )
    mats = np.array([np.eye(k), u.T @ bs @ u, u.T @ bn @ u])
    t = []
    v = []
    for p in range(pairs):
        teacher = np.random.default_rng([seed, 2, p]).normal(size=s)
        teacher *= np.sqrt(D) / np.linalg.norm(teacher)
        student = np.random.default_rng([seed, 3, p]).normal(size=k)
        student *= np.sqrt(k) / np.linalg.norm(student)
        t.append(u.T @ e[:, :s] @ teacher)
        v.append(u.T @ student)
    return np.array(t), np.array(v), ev, mats, bs, bn


def references(
    encoder, *, dynamics=None, rates=None, pairs=1, seed=0, checkpoints=None
):
    """Full annealed (one encoder) and conditioned (the same pairs as simulate) on CPU."""
    d = dynamics or Dynamics(D=encoder.weights.shape[1], policy="fixed")
    rates = _rates(d, rates)
    t, v, ev, mats, bs, bn = _actual(encoder, d, pairs, seed)
    cps = checkpoints or {}
    cp = cps.get("conditioned")
    start = cp["epoch"] if cp else 0
    k = len(ev)
    g = Geometry(
        bs, bn, np.full(k, 1 / d.D), 1 - encoder.signal_dim / d.D, encoder.delta
    )
    full = _modal(g, d, rates, cps.get("full"), "full_annealed")
    with jax.default_device(jax.devices("cpu")[0]):
        states, (mean, noise) = conditioned_modes(
            cp["mean"] if cp else v,
            t,
            ev,
            mats,
            D=d.D,
            epochs=d.epochs,
            switch=d.switch,
            T=d.T,
            epsilon=d.epsilon,
            eta=rates,
            sphere=d.spherical,
            schedule=("sft", "rl", "switch").index(d.schedule),
            initial_noise=cp["noise"] if cp else None,
            start_epoch=start,
            return_checkpoint=True,
        )
    conditioned = _result(
        states,
        rates,
        encoder.delta,
        "conditioned",
        dict(epoch=start + d.epochs, mean=np.asarray(mean), noise=np.asarray(noise)),
        "cpu",
        start,
    )
    return full, conditioned


def simulate(
    encoder: Encoder,
    *,
    dynamics=None,
    rates=None,
    pairs=1,
    repetitions=1,
    seed=0,
    checkpoint=None,
):
    """Actual microscopic runs on GPU; states have axes (epoch, pair, repetition, observable)."""
    d = dynamics or Dynamics(D=encoder.weights.shape[1], policy="fixed")
    try:
        device = jax.devices("gpu")[0]
    except (RuntimeError, IndexError) as exc:
        raise RuntimeError(
            "Microscopic simulation needs a CUDA GPU and the gpu install extra; use predict for CPU-only work."
        ) from exc
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    jax.config.update("jax_enable_x64", True)
    rates = _rates(d, rates)
    t, v, ev, mats, _, _ = _actual(encoder, d, pairs, seed)
    t = np.repeat(t, repetitions, axis=0)
    v = np.repeat(v, repetitions, axis=0)
    beta = t / np.sqrt(d.D * ev)
    remaining = 1 - np.sum(beta**2, axis=1)
    if remaining.min() < 0:
        raise ArithmeticError("Invalid residual teacher power")
    radius = np.sum(v * v, axis=1)
    start = checkpoint["epoch"] if checkpoint else 0
    seeds = [
        np.random.SeedSequence([seed, 4, p, r]).generate_state(1)[0]
        for p in range(pairs)
        for r in range(repetitions)
    ]
    with jax.default_device(device):
        vv = jnp.asarray(checkpoint["v"] if checkpoint else v)
        keys = (
            jnp.asarray(checkpoint["keys"])
            if checkpoint
            else jnp.stack([jax.random.PRNGKey(i) for i in seeds])
        )
        history = [observables(np.asarray(vv), t, mats, d.D)]
        for i, rr in enumerate(rates.reshape(d.epochs, d.D)):
            rl = d.schedule == "rl" or (
                d.schedule == "switch" and i + start >= d.switch
            )
            vv, keys = microscopic_epoch(
                vv,
                keys,
                np.broadcast_to(ev, v.shape),
                beta,
                np.sqrt(remaining),
                radius,
                D=d.D,
                T=d.T,
                epsilon=d.epsilon,
                eta=rr,
                rl=rl,
                sphere=d.spherical,
            )
            history.append(observables(np.asarray(vv), t, mats, d.D))
    return _result(
        np.asarray(history).reshape(d.epochs + 1, pairs, repetitions, 6),
        rates,
        encoder.delta,
        "microscopic",
        dict(epoch=start + d.epochs, v=np.asarray(vv), keys=np.asarray(keys)),
        "gpu",
        start,
    )
