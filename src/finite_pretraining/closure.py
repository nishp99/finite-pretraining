"""Fit and reuse one positive-time C5 geometry across pretraining times."""

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import jax
from threadpoolctl import threadpool_limits
from . import rmt
from ._io import write_json
from ._numerics import c5
from ._numerics.geometry import from_ha
from ._numerics.chains import TraceConvergenceError


@dataclass
class Fit:
    parameters: dict
    status: str
    nodes: np.ndarray | None
    overlap: np.ndarray | None
    diagnostics: dict

    def save(self, path):
        write_json(path, dict(format="finite-pretraining.fit.v1", **vars(self)))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.pop("format") != "finite-pretraining.fit.v1":
            raise ValueError("Unsupported fit format")
        for name in ("nodes", "overlap"):
            if data[name] is not None:
                data[name] = np.asarray(data[name], float)
        return cls(**data)

    def geometry(self, pretraining_time, *, allow_approximate=False):
        if self.status != "fit_converged" and not (
            allow_approximate and self.status == "physical_approximate"
        ):
            raise ValueError(
                f"Cannot predict with {self.status}; inspect fit.diagnostics"
            )
        p = self.parameters
        h, _ = c5.weights_and_derivative(self.nodes, [pretraining_time], p["q0"])
        return from_ha(h[0], self.overlap, p["phi"], p["rho"], p["delta"])


def fit_surrogate(
    alpha,
    *,
    phi=0.2,
    rho=0.8,
    delta=0.1,
    q0=0.05,
    kappa=0.0,
    m=24,
    training_times=(0.5, 2.0, 10.0, np.inf),
    objective="diag",
    initialization="current",
):
    """Cold fit from population parameters. No restarts, disk cache or empirical inputs."""
    rmt.validate_width(alpha, rho, delta, phi)
    if kappa != 0:
        raise ValueError(
            "C5 is validated for kappa=0 only; regularized LAE and RMT are available separately"
        )
    times = c5.positive_times(training_times)
    if len(np.unique(times)) != len(times):
        raise ValueError("Repeated fitting time")
    if not 0 < q0 < 1 or not isinstance(m, int) or m < 2:
        raise ValueError("Require 0<q0<1 and integer m>=2")
    if objective not in c5.OBJECTIVES:
        raise ValueError(f"Choose objective from {c5.OBJECTIVES}")
    if initialization not in ("current", "profile"):
        raise ValueError("Choose current or profile initialization")
    rank = min(m, int(np.floor((1 - rho) * m / phi + 1e-9)))
    ones = max(0, int(np.ceil((phi - rho) * m / phi - 1e-9)))
    if initialization == "profile" and (rank != m or ones):
        raise ValueError("Profile initialization requires full free signal rank")
    p = dict(
        alpha=alpha,
        phi=phi,
        rho=rho,
        delta=delta,
        q0=q0,
        kappa=kappa,
        m=m,
        training_times=times.tolist(),
        objective=objective,
        initialization=initialization,
    )
    report = {"traces": []}
    nodes = None
    if rank == 0 or rank < ones:
        return Fit(
            p,
            "unresolved_input",
            None,
            None,
            dict(reason="No admissible rank allocation"),
        )
    jax.config.update("jax_enable_x64", True)
    with jax.default_device(jax.devices("cpu")[0]), threadpool_limits(limits=1):
        try:
            for t in times:
                r = rmt.trace_statistics(alpha, t, phi=phi, rho=rho, delta=delta, q0=q0)
                report["traces"].append(
                    {
                        k: r[k]
                        for k in ("values", "absolute_error", "converged", "window")
                    }
                )
            targets = [r["values"] for r in report["traces"]]
            errors = [r["absolute_error"] for r in report["traces"]]
            grid, bands = rmt.calibration_grid(alpha, phi, rho, delta)
            cal = c5.calibrate_nodes(targets, times, phi, grid, bands, m, errors, q0)
            report["node_calibration"] = cal
            if not cal["valid_nodes"]:
                return Fit(p, "unresolved_input", None, None, report)
            nodes = cal["nodes"]
            profile = rmt.conditional_profile(nodes, alpha, rho, delta)
            report["profile"] = profile
            h, _ = c5.weights_and_derivative(nodes, times, q0)
            _, result = c5.solve_geometry(
                targets,
                h,
                phi,
                rho,
                delta,
                objective,
                errors,
                profile,
                initialization=initialization,
            )
        except (ArithmeticError, np.linalg.LinAlgError) as exc:
            report["reason"] = str(exc)
            if isinstance(exc, TraceConvergenceError):
                report["unresolved_trace"] = exc.diagnostic
            return Fit(p, "unresolved_input", nodes, None, report)
    report["solver"] = {
        k: result[k]
        for k in (
            "stage1_nfev",
            "stage1_message",
            "stage2_nit",
            "stage2_message",
            "stage2_success",
            "physical_valid",
            "trace_feasible",
            "candidate_origin",
            "maximum_tolerance_ratio",
            "normalized_residual",
            "residual_l2",
            "residual_rms",
            "objective",
            "physical_margins",
        )
        if k in result
    }
    status = result["status"]
    if status == "fit_converged" and not cal["feasible"]:
        status = "physical_approximate"
    return Fit(p, status, nodes, result.get("a"), report)
