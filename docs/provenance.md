# Source provenance

This repository starts a fresh history. Its numerical reference is the private [le-cake revision 4b491a90](https://github.com/nishp99/le-cake/tree/4b491a90a7415262850fd80173f9d7e3b22613e0).
It is not a runtime or installation dependency.

| Extracted component | Reference source |
|---|---|
| Aligned GD loss and spectral flow | linear_ae/train_autoencoder.py, linear_ae/finite_time_validation.py, sft_rl/weighted_traces.py |
| Population support, boundary and cubic chain | sft_rl/spectral_window.py, trace_expectations_v6.py, trace_expectations_v7.py, investigations/c5_fixed_width/spectral_audit.py |
| Node calibration, fit and geometry | sft_rl_finite_time/c5_positive_time.py, C5 parameterization from canonical_surrogate.py and Geometry from physical_surrogate.py |
| Downstream ODE, conditioned references and GPU updates | sft_rl_finite_time/annealed_surrogate_ode.py and investigations/physical_surrogate/ensemble_kernels.py |

The compact regression fixtures in tests/fixtures were extracted from the fixed-v1-20261101 delivery at that revision. alpha075.json contains the fitted A, nodes, profile, fit checks and four trace targets for alpha=0.75, phi=0.2, m=24, diagonal objective, current start. downstream.npz contains epochs 0, 5, 20 and 100 from g000 at ell=0.5, D=800, for six scenarios and fixed/adaptive rates. Only tests read these files. There are no raw training trajectories, production model weights or cluster-specific paths needed by the examples.
