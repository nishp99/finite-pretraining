# From a pretraining dataset to a learning curve

The first signal coordinates have unit population variance and the remaining noise coordinates have variance delta. Pretraining uses a tied linear encoder E: reconstruction is EᵀEx. Hidden width K is independent of signal dimension. The aligned initialization places the rows along the top K sample-covariance eigenvectors, with squared singular value q0. Full-batch GD uses the actual matrix gradient of the mean squared reconstruction error plus the unnormalized penalty λ‖E‖². It never projects the learned rows back onto the analytic solution.

For a selected sample eigenvalue x, the aligned flow satisfies **dq/dℓ = 2q[x(1−q)−κ]**, where κ=Dλ/2 and ℓ=4Ση/D. The spectral encoder rescales each eigenvector by √q. The trained and spectral encoders refer to the same sampled covariance; increasing the GD step count at fixed ℓ reduces discretization error. A random-initialized or minibatch optimizer is outside this model.

| Parameter | Meaning |
|---|---|
| D, alpha | Input dimension and N/D, where N is the number of pretraining examples |
| phi, rho, delta | K/D, noise fraction, and noise-to-signal variance ratio |
| q0, kappa, pretraining_time | Initial squared singular value, scaled L2 penalty and flow time ℓ |
| m | Surrogate dimension (default 24); distinct from K |
| training_times | Positive times used to calibrate a shared geometry, default 0.5, 2, 10, infinity |
| objective, initialization | Static C5 fit objective and prescribed solver start, independent of the downstream student's initialization |
| T, epsilon, eta | Samples per downstream update, SFT label-flip probability and fixed learning rate |
| schedule, switch | sft, rl or switch; switch counts completed SFT epochs |
| policy, spherical | fixed, adaptive or replay; optional fixed student norm |

RMT integrates the upper-phi spectral window to obtain nine D-normalized traces. Writing H=EEᵀ and B for the signal block of EEᵀ, these are tr(H), tr(H²), tr(H³), tr(B), tr(HB), tr(H²B), tr(B²), tr(HB²), tr(B³), each divided by D. One- and two-point spectral densities supply the first eight; a resolvent-chain contour integral supplies the cubic signal trace. Grid and contour refinement supply numerical error estimates. Near alpha=1 the support calculation retains the small positive edge; a spectral-window mass failure triggers only bounded refinement and an independent quadrature check.

C5 calibrates ordered covariance nodes to the three H moments across the fitting times, then fits one positive contraction A using the six remaining traces per time. At a new ℓ the node weights change but A does not. The default diagonal objective matches the conditional RMT signal profile in the pretraining basis. The full objective also penalizes off-diagonal entries. Available names: **diag**, **full**, **entropy**, and **entropy_diag_X** or **entropy_full_X** for X in 0.01, 0.1, 1, 10. Profile initialization is a deterministic 0.05 rotation of profile-based initial eigenvalues; it requires full free signal rank. It is optional, not an additional RMT identity.

Both solver stages retain the source budgets (1,500 least-squares evaluations and 2,000 constrained SLSQP iterations). A trace tolerance is max(1e−8, 2e−4 times the absolute target), enlarged for A-dependent traces by RMT numerical uncertainty. Rank, covariance positivity and teacher-power constraints are checked. A successful physical feasible terminal point takes precedence, followed by the physical feasible checkpoint with lowest objective, then the physical checkpoint with smallest maximum residual and L2 residual. Inspect `fit.status` and `fit.diagnostics`. A physical approximation requires explicit `allow_approximate=True` to predict. Missing physical geometry cannot be used. No downstream curves select a checkpoint.

C5 supports κ=0 and ℓ>0 (including positive infinity; use `"inf"` in JSON). Pretraining and RMT separately support κ≥0 and the aligned initial state. At alpha<1 require phi<alpha to avoid selecting sample-null modes. For an actual dataset N, K and signal dimension must be integral. Other admissible widths can be requested; a static fit need not resolve there.

The downstream teacher has norm √D in the signal subspace and each initial student has norm √K. One epoch contains D updates. The six observables J0, Js, Jn and Q0, Qs, Qn are D-normalized student–teacher and student–student contractions in the identity, signal and noise blocks. Accuracy is 1/2 + arcsin(J0/√(Qs+delta·Qn))/π. The modal ODE evolves on CPU. The adaptive rate maximizes the instantaneous accuracy-derivative quadratic over [0,1]; this is not a global optimal-control solution.

Use `Trajectory.checkpoint` and pass it back to the same model, settings and seeds with a new segment length to resume at the correct switch epoch. With replay supply only the rates for the new segment (one value per update). For references, pass `checkpoints={"full": full.checkpoint, "conditioned": conditioned.checkpoint}`. The full reference averages orientations for one actual encoder; the conditioned reference retains the specified teacher/student pairs. Microscopic repetitions keep those pairs fixed and use independent update randomness.
