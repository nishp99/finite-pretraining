from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from finite_pretraining import (
    Dynamics,
    Fit,
    predict,
    fit_surrogate,
    generate_dataset,
    spectral_encoder,
    train_lae,
    references,
)
from finite_pretraining import rmt
from finite_pretraining._numerics import c5

FIX = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("schedule", ["sft", "rl", "switch"])
@pytest.mark.parametrize("sphere", [False, True])
@pytest.mark.parametrize("policy", ["fixed", "adaptive"])
def test_frozen_dynamics(schedule, sphere, policy):
    fit = Fit.load(FIX / "alpha075.json")
    tr = predict(
        fit,
        0.5,
        dynamics=Dynamics(
            epochs=100, schedule=schedule, spherical=sphere, policy=policy
        ),
    )
    key = f"{schedule}_{'sphere' if sphere else 'free'}_fixed_{policy}"
    with np.load(FIX / "downstream.npz") as ref:
        np.testing.assert_allclose(
            tr.states[[0, 5, 20, 100]], ref[key], atol=2e-10, rtol=2e-10
        )
    assert tr.backend == "cpu"


def test_resume_and_roundtrip(tmp_path):
    fit = Fit.load(FIX / "alpha075.json")
    fit.save(tmp_path / "fit.json")
    d = Dynamics(D=80, epochs=4, switch=2)
    both = predict(fit, 0.5, dynamics=d)
    short = replace(d, epochs=2)
    first = predict(Fit.load(tmp_path / "fit.json"), 0.5, dynamics=short)
    last = predict(fit, 0.5, dynamics=short, checkpoint=first.checkpoint)
    np.testing.assert_allclose(last.states, both.states[2:], atol=1e-13)
    np.testing.assert_allclose(last.rates, both.rates[160:], atol=1e-13)
    replay = predict(fit, 0.5, dynamics=replace(d, policy="replay"), rates=both.rates)
    np.testing.assert_array_equal(replay.states, both.states)


def test_aligned_gd_refines_at_variable_width():
    data = generate_dataset(60, 1.5, seed=9)
    for phi in (0.1, 0.4):
        exact = spectral_encoder(data, 0.5, phi=phi, kappa=0.1)
        errors = [
            np.linalg.norm(
                train_lae(data, 0.5, steps=n, phi=phi, kappa=0.1).weights
                - exact.weights
            )
            for n in (80, 320)
        ]
        assert errors[1] < errors[0] / 3
        assert errors[1] / np.linalg.norm(exact.weights) < 0.004


def test_spectral_checks_and_status():
    for alpha in (1 - 1e-6, 1 + 1e-6, 0.5843886243):
        w = rmt.find_window(alpha, 0.8, 0.1, phi=0.2)
        assert abs(w["selected_mass"] - 0.2) < 2e-6
    with pytest.raises(ValueError, match="kappa"):
        fit_surrogate(0.75, kappa=0.1)
    with pytest.raises(ValueError, match="rank"):
        fit_surrogate(0.75, phi=0.4, initialization="profile")
    with pytest.raises(ValueError, match="ell"):
        predict(Fit.load(FIX / "alpha075.json"), 0)
    no_rank = fit_surrogate(2, phi=1, rho=0.99, m=2)
    assert no_rank.status == "unresolved_input"
    with pytest.raises(ValueError, match="unresolved_input"):
        predict(no_rank, 1, allow_approximate=True)
    approx = replace(Fit.load(FIX / "alpha075.json"), status="physical_approximate")
    with pytest.raises(ValueError, match="physical_approximate"):
        predict(approx, 1)
    assert predict(approx, 1, allow_approximate=True).status == "physical_approximate"


@pytest.mark.parametrize("method", c5.OBJECTIVES)
def test_objective_and_single_path_fit(method):
    # An exactly realizable set of moments with four modes. All objectives
    # receive identical contractions and constraints, including endpoint ranks.
    from finite_pretraining._numerics.geometry import from_ha

    n = np.array([0.3, 0.8, 1.5, 2.0])
    ts = [0.5, 2, 10, np.inf]
    h, _ = c5.weights_and_derivative(n, ts)
    p = np.array([0.2, 0.3, 0.5, 0.7])
    targets = np.array([from_ha(v, np.diag(p), 0.1).moments() for v in h])
    _, report = c5.solve_geometry(targets, h, 0.1, 0.8, 0.1, method, profile=p)
    assert (
        report["physical_valid"]
        and report["trace_feasible"]
        and np.isfinite(report["objective"])
    )


def test_conditioned_continuation():
    enc = spectral_encoder(generate_dataset(40, 1), 0.5)
    d = Dynamics(D=40, epochs=4, switch=2, policy="replay")
    rates = np.linspace(0.1, 0.9, 160)
    a = references(enc, dynamics=d, rates=rates, pairs=3)
    b = references(enc, dynamics=replace(d, epochs=2), rates=rates[:80], pairs=3)
    c = references(
        enc,
        dynamics=replace(d, epochs=2),
        rates=rates[80:],
        pairs=3,
        checkpoints=dict(full=b[0].checkpoint, conditioned=b[1].checkpoint),
    )
    for full, last in zip(a, c):
        np.testing.assert_allclose(full.states[2:], last.states, atol=1e-12)


@pytest.mark.cold
def test_cold_fit():
    original = Fit.load(FIX / "alpha075.json")
    new = fit_surrogate(0.75)
    assert new.status == original.status == "fit_converged"
    assert (
        new.diagnostics["node_calibration"]["feasible"]
        and new.diagnostics["solver"]["maximum_tolerance_ratio"] <= 1
    )
    np.testing.assert_allclose(
        [x["values"] for x in new.diagnostics["traces"]],
        [x["values"] for x in original.diagnostics["traces"]],
        atol=2e-10,
    )
    # Different LAPACK implementations can land at different feasible nodes.
    # Compare the three actual calibration moments at every training time.
    for f in [original, new]:
        h, _ = c5.weights_and_derivative(f.nodes, [0.5, 2, 10, np.inf])
        moments = 0.2 * np.stack([np.mean(h**k, axis=1) for k in (1, 2, 3)], axis=1)
        np.testing.assert_allclose(
            moments,
            np.array([x["values"][:3] for x in original.diagnostics["traces"]]),
            atol=1e-8,
            rtol=2e-4,
        )
