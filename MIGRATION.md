# Migrating from mlpp-lib 0.x to 1.0

mlpp-lib 1.0 replaces TensorFlow and TensorFlow Probability with
[Keras 3](https://keras.io) on the [PyTorch](https://pytorch.org) backend,
[`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) and
[scoringrules](https://frazane.github.io/scoringrules/). This is a breaking release.

The TensorFlow-based line is maintained on the `v0.x` branch (releases `0.16.x`).

## Before you upgrade

- **Saved models are not compatible.** Models trained with mlpp-lib < 1.0 (TensorFlow
  weights, `.h5` checkpoints or `.keras` files) cannot be loaded with mlpp-lib 1.0, not even
  by rebuilding the architecture and calling `load_weights`. Keep serving them with
  `mlpp-lib<1.0` until they are retrained with 1.0.
- **Python >= 3.10** is required.

## Installation and backend

```
# optional: CPU-only torch, to avoid downloading the CUDA build
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "mlpp-lib>=1.0"
```

`import mlpp_lib` sets `KERAS_BACKEND=torch` if it is not set. If you import `keras`
**before** `mlpp_lib`, set `KERAS_BACKEND=torch` in the environment yourself, otherwise
keras uses the backend from `~/.keras/keras.json` (often `tensorflow`) and mlpp-lib raises
an `ImportError`.

## Configurations

Most run configurations keep working, with deprecation warnings.

### Probabilistic layers

The distributions have new names. The old names are still accepted (with a
`DeprecationWarning`) and reproduce the old behaviour:

| mlpp-lib 0.x | mlpp-lib 1.0 |
|---|---|
| `IndependentNormal` | `Normal` |
| `IndependentLogistic` | `Logistic` |
| `IndependentLogNormal` | `LogNormal` |
| `IndependentTruncatedNormal` | `TruncatedNormal` (default bounds `low=0`, `high=inf`) |
| `IndependentDoublyCensoredNormal` | `CensoredNormal` with `{"low": 0, "high": 1}` (the default of `CensoredNormal` is `low=0`, `high=inf`) |
| `IndependentGamma` | `Gamma` |
| `IndependentBeta` | `Beta` |
| `IndependentWeibull` | `Weibull` |
| `IndependentPoisson` | `Poisson` |
| `IndependentBernoulli` | `Bernoulli` |
| `IndependentMixtureNormal` | `MixtureNormal` (`num_components=2`) |
| `MultivariateNormalDiag` | `Normal` (independent normal for each output) |
| `MultivariateNormalTriL` | `MultivariateNormalTriL` |
| `Independent4ParamsBeta`, `IndependentConcaveBeta`, `IndependentLogitNormal` | not available (yet) |

New: `Exponential`. See the README for the full list.

- Both `"probabilistic_layer": "Normal"` and `"probabilistic_layer": {"Normal": {options}}`
  are supported. Options can also be passed with `prob_layer_kwargs`.
- The TFP options `event_shape`, `convert_to_tensor_fn` and `validate_args` are ignored
  (with a warning). The number of output variables is always `output_size`.

### Losses

| mlpp-lib 0.x | mlpp-lib 1.0 |
|---|---|
| `crps_energy` | `CRPSEnsemble` (`crps_energy` still works, with `num_samples=1000`) |
| `WeightedCRPSEnergy(threshold, n_samples, correct_crps)` | `WeightedCRPSEnergy(threshold, num_samples, estimator)`. `n_samples` and `correct_crps` are deprecated but still work. Also see `TWCRPSEnsemble(a, b)`. |
| `EnergyScore(n_samples)` | `EnergyScore(num_samples)` (default 100 samples; `n_samples` is deprecated). The default `akr_circperm` estimator is unbiased and its memory grows linearly with the number of samples, like in 0.x. |
| `MultivariateLoss` | `MultivariateLoss` (for distributions, "mse"/"mae" use the mean) |
| `CombinedLoss` | `CombinedLoss` |
| `crps_energy_ensemble` | `CRPSEnsemble` with an ensemble tensor `y_pred`. Note that the members are now on axis 1 (`[batch, members, ...]`). |
| `MultiScaleCRPSEnergy`, `BinaryClassifierLoss` | not available (yet) |
| keras built-in losses | still available, e.g. `"mse"` |

New: closed-form CRPS losses (`CRPSNormal`, `CRPSTruncatedNormal`, `CRPSCensoredNormal`, ...),
`NegativeLogLikelihood`, and the `DistributionLossWrapper`/`SampleLossWrapper` wrappers
for any scoringrules function.

All losses now ignore missing (NaN) targets. mlpp-lib 0.x raised an error instead.

### Metrics

`bias` is an alias of `expected_bias`. `EnsembleMetrics` uses scoringrules with the same
values as the (biased) CRPS estimator of properscoring, so they stay comparable. It
predicts in batches (`batch_size`, default 100000) to limit memory.

### Models

- The model builders in `mlpp_lib.models` no longer take `input_shape` as first argument:
  `fully_connected_network(output_size, hidden_layers, ...)`. `mlpp_lib.utils.get_model`
  still takes `input_shape` and returns a built model.
- Probabilistic models are `ProbabilisticModel`s. Calling them returns a wrapped
  `torch.distributions.Distribution`, with `.mean`, `.sample(n)` (shape `[n, batch, outputs]`),
  `.log_prob(y)`, etc. Use `model(x, output_type="samples")` or `output_type="expected"` to get
  samples or the mean directly.
- `fully_connected_multibranch_network`: `n_branches` is optional and still defaults to one
  branch per distribution parameter. `aggregation` ("concat" or "sum") is new.
- `deep_cross_network`: the new optional arguments `n_cross_layers` (default
  `len(hidden_layers)`) and `cross_layers_hiddensize` (default: no projection of the inputs)
  are available. The cross layers now use the original DCN formulation
  x_{l+1} = x_0 (x_l^T w_l) + b_l + x_l, and `skip_connection` now applies to the deep part.
- `skip_connection` in the fully connected networks now adds a linear projection of the
  inputs to the output of the last hidden layer.
- The temporal convolutional network (`temporal_convolutional_network`), the physical layers
  (`mlpp_lib.physical_layers`, `ThermodynamicLayer`) and the models using them
  (`architecture_constrained_fcn`, `architecture_constrained_tcn`) have been removed.

### Training

- `out_bias_init: "mean"` now initializes the bias with the mean of the **targets**
  (in 0.x it used the mean of the features).
- `DataLoader` keeps the data on the CPU and moves each batch to the device. It has a new
  `seed` argument.
