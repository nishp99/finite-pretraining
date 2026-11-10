"""Command-line paths share the public Python API."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from . import (
    Dynamics,
    Fit,
    fit_surrogate,
    generate_dataset,
    predict,
    references,
    simulate,
    spectral_encoder,
    train_lae,
)
from ._io import write_json
from .finetuning import OBSERVABLES
from .rmt import encoder_weighted_traces


def save_trajectory(path, result):
    np.savez_compressed(
        path,
        epoch=result.epoch,
        states=result.states,
        accuracy=result.accuracy,
        rates=result.rates,
        observable_names=OBSERVABLES,
        status=result.status,
        backend=result.backend,
        **{"checkpoint_" + k: v for k, v in result.checkpoint.items()},
    )


def run(command, config_path, output):
    cfg = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Choose an empty output directory: {output}")
    model = cfg.pop("model", {})
    fit_file = cfg.pop("fit_file", None)
    if fit_file and model:
        raise ValueError("Specify either a saved fit_file or model parameters")
    d = Dynamics(**cfg.pop("dynamics", {}))
    ell = float(cfg.pop("pretraining_time"))
    approx = cfg.pop("allow_approximate", False)
    ens = cfg.pop("ensemble", {})
    rate_file = cfg.pop("rates_file", None)
    if cfg:
        raise ValueError(f"Unknown config fields: {list(cfg)}")
    if not ell > 0:
        raise ValueError("C5 prediction requires positive pretraining_time")
    fit = (
        Fit.load(config_path.parent / fit_file) if fit_file else fit_surrogate(**model)
    )
    fit.save(output / "fit.json")
    rates = (
        np.load(config_path.parent / rate_file, allow_pickle=False)
        if rate_file
        else None
    )
    counts = dict(encoders=1, pairs=2, repetitions=2, steps=500, seed=17)
    if set(ens) - counts.keys():
        raise ValueError("Unknown ensemble settings")
    counts.update(ens)
    if any(
        not isinstance(counts[k], int) or counts[k] < 1
        for k in ("encoders", "pairs", "repetitions", "steps")
    ):
        raise ValueError("Ensemble counts and GD steps must be positive integers")
    resolved = dict(
        command=command,
        model=fit.parameters,
        dynamics=asdict(d),
        pretraining_time=ell,
        allow_approximate=approx,
        fit_source="saved" if fit_file else "computed",
        ensemble=counts if command == "compare" else None,
    )
    write_json(output / "config.json", resolved)
    pred = predict(fit, ell, dynamics=d, allow_approximate=approx, rates=rates)
    save_trajectory(output / "prediction.npz", pred)
    if command == "predict":
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = fit.parameters
    records = []
    curves = []
    for i in range(counts["encoders"]):
        seed = int(np.random.SeedSequence([counts["seed"], i]).generate_state(1)[0])
        data = generate_dataset(
            d.D, p["alpha"], rho=p["rho"], delta=p["delta"], seed=seed
        )
        options = dict(phi=p["phi"], q0=p["q0"], kappa=p["kappa"])
        enc = train_lae(data, ell, steps=counts["steps"], **options)
        analytic = spectral_encoder(data, ell, **options)
        actual_trace = np.array(encoder_weighted_traces(enc.weights, enc.signal_dim))
        exact_trace = np.array(
            encoder_weighted_traces(analytic.weights, enc.signal_dim)
        )
        records.append(
            dict(
                encoder=i,
                seed=seed,
                flow_time=enc.pretraining_time,
                learning_rate=ell * d.D / (4 * counts["steps"]),
                actual_traces=actual_trace,
                spectral_traces=exact_trace,
                relative_encoder_error=np.linalg.norm(enc.weights - analytic.weights)
                / np.linalg.norm(analytic.weights),
            )
        )
        opts = dict(dynamics=d, rates=pred.rates, pairs=counts["pairs"], seed=seed)
        sim = simulate(enc, repetitions=counts["repetitions"], **opts)
        full, conditioned = references(enc, **opts)
        for label, tr in [("gpu", sim), ("full", full), ("conditioned", conditioned)]:
            save_trajectory(output / f"encoder-{i}-{label}.npz", tr)
        curves.append(
            [
                sim.accuracy.mean(axis=(1, 2)),
                full.accuracy,
                conditioned.accuracy.mean(axis=1),
            ]
        )
    write_json(output / "pretraining-check.json", records)
    fig, ax = plt.subplots(layout="constrained")
    labels = [
        "GPU: learned GD encoders",
        "Full annealed: same encoders",
        "Conditioned: same initial pairs",
    ]
    for y, label in zip(np.mean(curves, axis=0), labels):
        ax.plot(pred.epoch, 100 * y, label=label)
    ax.plot(pred.epoch, 100 * pred.accuracy, "--", label="C5: RMT inputs")
    ax.set(
        title=f"{d.schedule}, {'spherical' if d.spherical else 'free'}; {d.policy} rates replayed",
        xlabel="Downstream epochs",
        ylabel="Accuracy (%)",
    )
    ax.legend()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("predict", "compare"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    try:
        run(a.command, a.config, a.output)
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        parser.exit(1, f"{exc}\n")
