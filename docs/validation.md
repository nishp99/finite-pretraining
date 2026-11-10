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
