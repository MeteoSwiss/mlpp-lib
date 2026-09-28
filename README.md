# mlpp-lib

Collection of methods for ML-based postprocessing of weather forecasts.

Since version 1.0, mlpp-lib is built on [Keras 3](https://keras.io) with the [PyTorch](https://pytorch.org) backend, uses [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) for its probabilistic layers and [scoringrules](https://frazane.github.io/scoringrules/) for its loss functions. See [MIGRATION.md](MIGRATION.md) if you are upgrading from mlpp-lib < 1.0 (tensorflow).

:warning: **The code in this repository is currently work-in-progress and not recommended for production use.** :warning:

## Installation

```
pip install mlpp-lib
```

To avoid downloading the (large) CUDA build of PyTorch on machines without a GPU, install the CPU build of torch first:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mlpp-lib
```


# Quickstart


```python
import numpy as np 
import xarray as xr 
import pandas as pd

from mlpp_lib.datasets import DataModule, DataSplitter
```


```python
LEADTIMES = np.arange(24)
REFTIMES = pd.date_range("2018-01-01", "2018-03-31", freq="24h")
STATIONS = [chr(i) * 3 for i in range(ord("A"), ord("Z"))]
SHAPE = (len(REFTIMES), len(LEADTIMES), len(STATIONS))
DIMS = ["forecast_reference_time", "lead_time", "station"]

def features_dataset() -> xr.Dataset:
    rng = np.random.default_rng(1)
    X = rng.standard_normal(size=(*SHAPE, 4))
    X[(X > 4.5) | (X < -4.5)] = np.nan

    features = xr.Dataset(
        {
            "coe:x1": (DIMS, X[..., 0]),
            "coe:x2": (DIMS, X[..., 1]),
            "obs:x3": (DIMS, X[..., 2]),
            "dem:x4": (DIMS, X[..., 3]),
        },
        coords={
            "forecast_reference_time": REFTIMES,
            "lead_time": LEADTIMES,
            "station": STATIONS,
        },
    )

    return features

def targets_dataset() -> xr.Dataset:
    """
    Create a dataset as if it was loaded from `targets.zarr`.
    """
    rng = np.random.default_rng(1)
    Y = rng.standard_normal(size=(*SHAPE, 2))
    Y[(Y > 4.5) | (Y < -4.5)] = np.nan

    targets = xr.Dataset(
        {"obs:y1": (DIMS, Y[..., 0]), "obs:y2": (DIMS, Y[..., 1])},
        coords={
            "forecast_reference_time": REFTIMES,
            "lead_time": LEADTIMES,
            "station": STATIONS,
        },
    )

    return targets


```

MLPP expects xarray objects that look like this:


```python
features = features_dataset()
print(features)

```

    <xarray.Dataset> Size: 2MB
    Dimensions:                  (forecast_reference_time: 90, lead_time: 24,
                                  station: 25)
    Coordinates:
      * forecast_reference_time  (forecast_reference_time) datetime64[us] 720B 20...
      * lead_time                (lead_time) int64 192B 0 1 2 3 4 ... 19 20 21 22 23
      * station                  (station) <U3 300B 'AAA' 'BBB' ... 'XXX' 'YYY'
    Data variables:
        coe:x1                   (forecast_reference_time, lead_time, station) float64 432kB ...
        coe:x2                   (forecast_reference_time, lead_time, station) float64 432kB ...
        obs:x3                   (forecast_reference_time, lead_time, station) float64 432kB ...
        dem:x4                   (forecast_reference_time, lead_time, station) float64 432kB ...



```python
targets = targets_dataset()
print(targets)
```

    <xarray.Dataset> Size: 865kB
    Dimensions:                  (forecast_reference_time: 90, lead_time: 24,
                                  station: 25)
    Coordinates:
      * forecast_reference_time  (forecast_reference_time) datetime64[us] 720B 20...
      * lead_time                (lead_time) int64 192B 0 1 2 3 4 ... 19 20 21 22 23
      * station                  (station) <U3 300B 'AAA' 'BBB' ... 'XXX' 'YYY'
    Data variables:
        obs:y1                   (forecast_reference_time, lead_time, station) float64 432kB ...
        obs:y2                   (forecast_reference_time, lead_time, station) float64 432kB ...


## Preparing data

The entire data processing can be handled by the `DataModule` class: 
- loading the raw data
- train, val, test splits
- normalization
- reshaping to a tensor


```python
splitter = DataSplitter(
    time_split={"train": 0.6, "val": 0.2, "test": 0.2},
    station_split={"train": 0.7, "val": 0.1, "test": 0.2},
    time_split_method="sequential",
    station_split_method="random",
)

datamodule = DataModule(
    features, targets[["obs:y1"]],
    batch_dims=["forecast_reference_time", "lead_time", "station"],
    splitter=splitter
)

datamodule.setup(stage=None)
```

    No normalizer found, data are standardized by default.


## Training
The library builds on top of PyTorch + Keras 3 API and provides some useful methods to quickly build probabilistic models, while integrating probabilistic metrics thanks to `scoringrules`. Of course, you're free to use torch and torch distributions to build your own custom model. MLPP won't get in your way!

In the following example the model consists of a fully connected layer and a probabilistic layer modelling a normal distribution parametrized by some predicted parameters, optimized with a closed form CRPS.


```python
import mlpp_lib  # sets the torch backend of keras
import keras

from mlpp_lib.layers import MultilayerPerceptron
from mlpp_lib.losses import CRPSNormal
from mlpp_lib.models import ProbabilisticModel
from mlpp_lib.probabilistic_layers import DistributionLayer

encoder = MultilayerPerceptron(
    hidden_layers=[16, 8],
    dropout=0.1,
    activations="sigmoid",
)
prob_layer = DistributionLayer("Normal")

model = ProbabilisticModel(encoder=encoder, output_distribution=prob_layer)
model.compile(loss=CRPSNormal(), optimizer=keras.optimizers.Adam(learning_rate=0.1))

history = model.fit(
    datamodule.train.x,
    datamodule.train.y,
    epochs=2,
    batch_size=32,
    validation_data=(datamodule.val.x, datamodule.val.y),
    verbose=2,
)
```

    Epoch 1/2


    689/689 - 4s - 5ms/step - loss: 0.5644 - val_loss: 0.5565


    Epoch 2/2


    689/689 - 4s - 7ms/step - loss: 0.5631 - val_loss: 0.5527


The same model can be built with `mlpp_lib.models.fully_connected_network`, as done by `mlpp_lib.train.train` from a configuration:


```python
from mlpp_lib.models import fully_connected_network

model = fully_connected_network(
    output_size=1,
    hidden_layers=[16, 8],
    dropout=0.1,
    activations="sigmoid",
    probabilistic_layer="Normal",
)
```

## Predictions
Once your model is trained, you can make predictions and create ensembles by sampling from the predictive distribution. The `Dataset` class comes with a method to wrap your ensemble predictions in a xarray object with the correct dimensions and coordinates.


```python
test_pred_ensemble = model(datamodule.test.x).sample(21).numpy()
test_pred_ensemble = datamodule.test.dataset_from_predictions(test_pred_ensemble, ensemble_axis=0)
print(test_pred_ensemble)
```

    <xarray.Dataset> Size: 363kB
    Dimensions:                  (realization: 21, forecast_reference_time: 18,
                                  lead_time: 24, station: 5)
    Coordinates:
      * realization              (realization) int64 168B 0 1 2 3 4 ... 17 18 19 20
      * forecast_reference_time  (forecast_reference_time) datetime64[us] 144B 20...
      * lead_time                (lead_time) int64 192B 0 1 2 3 4 ... 19 20 21 22 23
      * station                  (station) <U3 60B 'AAA' 'III' 'NNN' 'VVV' 'YYY'
    Data variables:
        obs:y1                   (realization, forecast_reference_time, lead_time, station) float64 363kB ...


# Predictive Distributions

Many predictive distributions are supported out-of-the-box and ready to be used in your model, by their name. All of them (but `MultivariateNormalTriL`) model each of the `output_size` target variables independently.

| Name | Comment | Reparametrized sampling | Closed form CRPS |
|------------|------------|:------------:|:------------:|
| `Normal` | | ✔️ | `CRPSNormal` |
| `Logistic` | | ✔️ | `CRPSLogistic` |
| `LogNormal` | | ✔️ | `CRPSLogNormal` |
| `TruncatedNormal` | A normal distribution with support bounded to $[low, high]$ (default $[0, \infty]$). | ✔️ | `CRPSTruncatedNormal` |
| `CensoredNormal` | A normal distribution where values outside $[low, high]$ are assigned to $low$ and $high$ (default $[0, \infty]$). | ✔️ | `CRPSCensoredNormal` |
| `Gamma` | | ✔️ | |
| `Beta` | | ✔️ | |
| `Weibull` | | ✔️ | |
| `Exponential` | | ✔️ | `CRPSExponential` |
| `Poisson` | | | `CRPSPoisson` |
| `Bernoulli` | | | |
| `MixtureNormal` | A mixture of `num_components` (default 2) normal distributions. | | `CRPSMixtureNormal` |
| `MultivariateNormalTriL` | A multivariate normal distribution, parametrized by the Cholesky factor of its covariance matrix. | ✔️ | |

Distribution options are passed with `distribution_kwargs`, e.g. `DistributionLayer("TruncatedNormal", distribution_kwargs={"low": -1.0})`, or with `prob_layer_kwargs` in the model builders.

Any other distribution available in [PyTorch](https://pytorch.org/docs/stable/distributions.html) can be introduced in our framework with little effort:
- Extend the `BaseParametricDistribution` class.
- Implement `BaseParametricDistribution.base_distribution` to enforce constraints on the parameters (e.g make sure they stay positive).
- Add a `DistributionLayer` layer to your model, instantiated with your new distribution. It will make sure to transform its input data into the expected number of parameters.

# Loss functions

Depending on the predictive distribution, the model can be optimized via a closed form CRPS, a sample-based CRPS or its log-likelihood.

In `mlpp` there are:
- Named closed-form losses such as `CRPSNormal`, `CRPSTruncatedNormal`, ... (see the table above).
- Named sample-based losses such as `CRPSEnsemble`, `TWCRPSEnsemble` (threshold-weighted CRPS) and `EnergyScore` (for multivariate distributions).
- `NegativeLogLikelihood`, `MultivariateLoss` and `CombinedLoss`.
- Wrappers for external modules, implemented in `DistributionLossWrapper` and `SampleLossWrapper`, for closed form and sample-based scores respectively.

For sample-based losses, the underlying distribution needs to have a reparametrized sampling function. If that was not available, `SampleLossWrapper` will let you know.

All losses ignore missing (NaN) target values.

Currently, `mlpp` relies on [scoringrules](https://frazane.github.io/scoringrules/) for its loss functions.


```python
import scoringrules as sr

from mlpp_lib.losses import (
    CRPSEnsemble,
    CRPSNormal,
    DistributionLossWrapper,
    NegativeLogLikelihood,
    SampleLossWrapper,
)

# Named losses
loss = CRPSNormal()
loss = CRPSEnsemble(num_samples=100)
loss = NegativeLogLikelihood()
# Closed form loss
loss = DistributionLossWrapper(fn=sr.crps_normal)
# Sample-based loss
loss = SampleLossWrapper(fn=sr.crps_ensemble, num_samples=100)
```

# Models

Probabilistic models return by default an object representing the parametric predictive distribution, which is used internally during the model optimization. It is however also possible to directly obtain a certain number of samples from the distribution or its expected value.


```python
encoder = MultilayerPerceptron(hidden_layers=[16, 8], activations="sigmoid")
prob_layer = DistributionLayer("Normal", event_size=2)

model = ProbabilisticModel(encoder=encoder, output_distribution=prob_layer)

inputs = keras.random.uniform((32, 5))

output_distribution = model(inputs)
print(f"Output distribution is a: {output_distribution} \n")

output_samples = model(inputs, output_type="samples", num_samples=21)
print(f"Output samples of shape: {output_samples.shape} \n")

expected_output = model(inputs, output_type="expected")
print(f"Expected output of shape: {expected_output.shape}")
```

    Output distribution is a: WrappingTorchDist(Independent(Normal(loc: torch.Size([32, 2]), scale: torch.Size([32, 2])), 1)) 
    
    Output samples of shape: torch.Size([21, 32, 2]) 
    
    Expected output of shape: torch.Size([32, 2])


## Build the README

```
poetry run jupyter nbconvert --execute --to markdown README.ipynb
```
