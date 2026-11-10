# Validation and practical limits

The reference fixed-width study used D=800, phi=0.2, rho=0.8, delta=0.1, kappa=0 and alpha from 0.5 to 4. It found 274/274 distinct nominal fits physical and converged and compared against **14,592 actual stochastic learning trajectories**. The full report is [available at the pinned source](https://github.com/nishp99/le-cake/blob/4b491a90/investigations/c5_fixed_width/RESULTS.md). This migration uses small numerical regression fixtures and a fresh comparison; it does not rerun that historical ensemble.

- Current and profile-informed starts both converged, but differed by up to 0.978 percentage points under fixed rates. An independently prescribed initialization makes the algorithm reproducible; it does not prove the geometry unique.
- Worst epoch-100 error across nine new settings was 4.037 percentage points for adaptive spherical RL. Much of that case's error comes from averaging teacher/student orientations. The full-annealed reference itself had a 3.984-point error. Comparing only the surrogate with the full reference would miss this discrepancy.
- More pretraining need not improve fixed-budget accuracy: adaptive spherical SFT decreased by 0.360 points in a matched GPU comparison. Only 358/600 long-time ODE chains met the stated plateau criteria by epoch 3,200.
- The dense validation and spectral-window repair cover the tested fixed-width domain. Other widths can cross rank transitions or give unresolved fits. For precision optimization, check dimension, initialization and actual candidate settings.

## Run the checks

```sh
python -m pip install '.[test]'
python -m pytest -m 'not gpu'             # includes cold RMT and fit
python -m pytest -m 'not gpu and not cold' # faster CPU regressions
python -m pytest -m gpu                   # on a CUDA worker
```

Saved-geometry trajectory tests are tight numerical regressions against the source. Cold fits also check the RMT values, calibrated moments and feasibility: equivalent physical constraints do not force different BLAS/optimizer implementations to return identical nodes. The GPU test independently computes supplied-draw updates and checks varying-rate checkpoint/restart across the SFT→RL switch. In comparisons, replay uses the surrogate's schedule; stochastic students do not each run an independent optimal controller. Ensemble means average each trajectory's accuracy, not an accuracy reconstructed from averaged states.

A fresh calculation at alpha 0.75 and width 0.2 was also compared directly with revision `4b491a90` in the same macOS CPU environment. Calibrated nodes agreed exactly, and the largest difference in the six downstream observables was below 2.4e-14.

## Standalone release verification

The CPU quickstart builds the RMT inputs from parameters with an empty cache. Loading its saved fit reproduced all six trajectories of observables exactly in the clean macOS wheel installation. A second cold calculation with width 0.1 also converged. The Linux CPU suite passed all 28 tests, including near-one spectral support, variable width, each objective, all six downstream scenarios, fixed and adaptive rates, and checkpoint continuation. [CPU verification run](https://github.com/nishp99/finite-pretraining/actions/runs/42924350416)

The example comparison ran in `jax_saad_solla` with a CPU for the cold fit and reference ODEs and an NVIDIA L40S for microscopic updates. The supplied-draw GPU update agreed with an independent NumPy calculation. Restarting a varying-rate run at the SFT→RL switch reproduced the uninterrupted states and random keys.

| Curve at epoch 20 | Accuracy |
|---|---:|
| C5 from population parameters | 90.94% |
| GPU learning on GD-trained encoders | 90.50% |
| Full annealed, averaged over those encoders | 90.65% |
| Conditioned on the same encoders and initialization pairs | 90.53% |

The example used D=200, alpha=0.75, phi=0.2 and pretraining time 0.5. Two encoders, two teacher/student pairs per encoder and two downstream repetitions produced eight trajectories. Each curve used the same 4,000 rates. The largest difference between C5 and the GPU mean across the 21 recorded epochs was 0.90 percentage points. Relative Frobenius errors between the two GD-trained encoders and their own spectral solutions were 0.0280% and 0.0286%.
