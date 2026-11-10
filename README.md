# Finite pretraining

Predict how a tied linear autoencoder trained for a finite time affects downstream supervised fine-tuning (SFT), reinforcement learning (RL), or SFT followed by RL. You can also train an actual encoder and compare its microscopic learning curve with the prediction.

## Install and predict on CPU

From this repository, with Python 3.11:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install .
finite-pretraining predict examples/predict.json --output runs/prediction
```

The first command-line prediction **computes a fit from population parameters**. Allow several minutes for spectral integration, constrained optimization and JAX compilation. It does not download models or use any old experiment outputs. The example uses alpha=0.75, width fraction 0.2, 200 input dimensions, spherical SFT switching to RL at epoch 10, and the bounded locally optimal rate.

```python
from finite_pretraining import Dynamics, Fit, predict

fit = Fit.load("runs/prediction/fit.json")
curve = predict(fit, pretraining_time=2, dynamics=Dynamics(D=200, epochs=20))
print(curve.epoch[-1], 100 * curve.accuracy[-1])
print(curve.observables["Js"][-1])
```

To fit directly in Python, use `fit_surrogate(alpha=0.75, phi=0.2)`. Reuse that fit for other positive pretraining times or downstream budgets; changing alpha or width requires a new fit. Accuracy is a fraction. `prediction.npz` stores epoch, six labeled observables, rates, accuracy and the final modal checkpoint. Output directories must be empty. Read saved NPZ files with `numpy.load(..., allow_pickle=False)`.

## Compare against actual learning on GPU

On a Linux CUDA machine, install `python -m pip install '.[gpu]'` (or install the package without replacing dependencies in an already compatible CUDA/JAX environment). First run the CPU prediction above, then:

```sh
finite-pretraining compare examples/compare.json --output runs/comparison
```

The comparison config explicitly points to the fit you just computed, relative to the config file. Alternatively replace `fit_file` with `"model": {"alpha": 0.75, "phi": 0.2}` to fit from scratch. The command generates two independent datasets, trains a full matrix of encoder weights by aligned full-batch GD, and checks each learned encoder against the analytic solution **on its own dataset**. The GPU fine-tunes the learned weights. It never substitutes the analytic encoder for the trained one. Change the encoder, pair and repetition counts in the example config.

`comparison.png` shows the RMT surrogate, the GPU ensemble mean, and full-annealed and pair-conditioned CPU references using the same encoders. All replay the same per-update rate schedule. `pretraining-check.json` reports encoder and trace errors; the NPZ files retain separate epoch, pair and repetition axes. This small example demonstrates the workflow; it does not estimate dataset-level uncertainty reliably.

## Python entry points

| Module | Start here |
|---|---|
| `pretraining` | `generate_dataset`, `spectral_encoder`, `train_lae` |
| `rmt` | `trace_statistics`, `find_window`, `conditional_profile` |
| `closure` | `fit_surrogate`, `Fit.save`, `Fit.load` |
| `finetuning` | `Dynamics`, `predict`, `simulate`, `references` |

```python
from finite_pretraining import Dynamics, Fit, generate_dataset, predict, references, simulate, train_lae

fit = Fit.load("runs/prediction/fit.json")

data = generate_dataset(D=200, alpha=0.75, seed=12)
encoder = train_lae(data, pretraining_time=0.5, phi=0.2, steps=500)
d = Dynamics(D=200, epochs=20, schedule="switch", switch=10, policy="adaptive")
rates = predict(fit, 0.5, dynamics=d).rates
actual = simulate(encoder, dynamics=d, rates=rates, pairs=2, repetitions=2, seed=73)
full, conditioned = references(encoder, dynamics=d, rates=rates, pairs=2, seed=73)
```

A CPU-only installation supports pretraining, RMT, fitting and both reference ODEs. Microscopic simulation requires GPU. GPU nodes must expose **both** CPU and CUDA JAX backends for the combined workflow.

See [methods and parameter glossary](docs/methods.md), [validation and limitations](docs/validation.md), and [source provenance](docs/provenance.md).
