"""Run on a CUDA worker: pytest -m gpu. A requested GPU test never falls back."""

from dataclasses import replace
import jax
import numpy as np
import pytest
from finite_pretraining import Dynamics, generate_dataset, references, simulate, train_lae
from finite_pretraining._numerics.microscopic import update_from_draws

pytestmark = pytest.mark.gpu


def test_supplied_draws_and_restart():
    jax.config.update("jax_enable_x64", True)
    gpu = jax.devices("gpu")[0]
    rng = np.random.default_rng(712)
    D = 40
    T = 7
    v = rng.normal(size=8)
    ev = np.linspace(0.1, 1, 8)
    beta = np.full(8, 0.12)
    remaining = np.sqrt(1 - beta @ beta)
    with jax.default_device(gpu):
        supplied_update = jax.jit(update_from_draws, static_argnums=(6, 7))
        for rl in (True, False):
            for sphere in (True, False):
                xi = rng.normal(size=(T, 9))
                flips = rng.uniform(size=T)
                z = xi[:, :-1] * np.sqrt(ev)
                target = np.sign(xi[:, :-1] @ beta + xi[:, -1] * remaining)
                pred = np.where(z @ v >= 0, 1, -1)
                labels = np.where(flips < 0.05, -target, target)
                coeff = pred * all(pred == target) if rl else (labels - pred) / 2
                w = v + 0.38 * (coeff @ z) / (T * np.sqrt(D))
                if sphere:
                    w *= np.linalg.norm(v) / np.linalg.norm(w)
                actual = supplied_update(
                    v,
                    ev,
                    beta,
                    remaining,
                    xi,
                    flips,
                    D,
                    T,
                    0.05,
                    0.38,
                    rl,
                    sphere,
                    v @ v,
                )
                assert actual.devices() == {gpu}
                np.testing.assert_allclose(w, actual, atol=1e-13)
    enc = train_lae(generate_dataset(D, 0.75), 0.5)
    d = Dynamics(D=D, epochs=2, switch=1, policy="replay")
    rates = np.linspace(0.1, 0.8, 2 * D)
    for sphere in (True, False):
        opts = dict(pairs=2, repetitions=2, seed=11)
        both = simulate(enc, dynamics=replace(d, spherical=sphere), rates=rates, **opts)
        one = replace(d, epochs=1, spherical=sphere)
        a = simulate(enc, dynamics=one, rates=rates[:D], **opts)
        b = simulate(
            enc, dynamics=one, rates=rates[D:], checkpoint=a.checkpoint, **opts
        )
        np.testing.assert_array_equal(b.states, both.states[1:])
        np.testing.assert_array_equal(b.checkpoint["keys"], both.checkpoint["keys"])
        full, conditioned = references(
            enc, dynamics=replace(d, spherical=sphere), rates=rates, pairs=2, seed=11
        )
        assert full.backend == conditioned.backend == "cpu" and both.backend == "gpu"
        np.testing.assert_allclose(
            both.states[0],
            np.repeat(conditioned.states[0, :, None], 2, axis=1),
            atol=1e-12,
        )
