# mlpp-lib

Collection of methods for ML-based postprocessing of weather forecasts.

:warning: **The code in this repository is currently work-in-progress and not recommended for production use.** :warning:



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
      * forecast_reference_time  (forecast_reference_time) datetime64[ns] 720B 20...
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
      * forecast_reference_time  (forecast_reference_time) datetime64[ns] 720B 20...
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
The library builds on top of PyTorch + Keras3 API and provides some useful methods to quickly build probabilistic models, while integrating probabilistic metrics thanks to `scoringrules`. Of course, you're free to use torch and torch distributions to build your own custom model. MLPP won't get in your way!

In the following example the model consists of a fully connected layer and a probabilistic layer modelling a normal distribution parametrized by some predicted parameters, which can either be optimized via a closed form CRPS or a sample-based CRPS.

For sample-based losses, the underlying distribution needs to have a reparametrized sampling function. If that was not available, `SampleLossWrapper` will let you know.


```python
from mlpp_lib.layers import FullyConnectedLayer
from mlpp_lib.models import ProbabilisticModel
from mlpp_lib.losses import DistributionLossWrapper, SampleLossWrapper
from mlpp_lib.probabilistic_layers import BaseDistributionLayer, UniveriateGaussianModule
import scoringrules as sr
import keras


encoder = FullyConnectedLayer(hidden_layers=[16,8], 
                                batchnorm=False, 
                                skip_connection=False,
                                dropout=0.1,
                                mc_dropout=False,
                                activations='sigmoid')
prob_layer = BaseDistributionLayer(distribution=UniveriateGaussianModule())

model = ProbabilisticModel(encoder_layer=encoder, probabilistic_layer=prob_layer)

# crps_normal = DistributionLossWrapper(fn=sr.crps_normal) # closed form CRPS
crps_normal = SampleLossWrapper(fn=sr.crps_ensemble, num_samples=100) # sample-based CRPS 

model.compile(loss=crps_normal, optimizer=keras.optimizers.Adam(learning_rate=0.1))

history = model.fit(
    datamodule.train.x, datamodule.train.y,
    epochs = 2,
    batch_size = 32,
    validation_data = (datamodule.val.x, datamodule.val.y)
)
```



## Predictions
Once your model is trained, you can make predictions and create ensembles by sampling from the predictive distribution. The `Dataset` class comes with a method to wrap your ensemble predictions in a xarray object with the correct dimensions and coordinates.


```python
test_pred_ensemble = model(datamodule.test.x).sample(21)
test_pred_ensemble = datamodule.test.dataset_from_predictions(test_pred_ensemble, ensemble_axis=0)
print(test_pred_ensemble)
```

    <xarray.Dataset> Size: 363kB
    Dimensions:                  (realization: 21, forecast_reference_time: 18,
                                  lead_time: 24, station: 5)
    Coordinates:
      * forecast_reference_time  (forecast_reference_time) datetime64[ns] 144B 20...
      * lead_time                (lead_time) int64 192B 0 1 2 3 4 ... 19 20 21 22 23
      * station                  (station) <U3 60B 'AAA' 'III' 'NNN' 'VVV' 'YYY'
      * realization              (realization) int64 168B 0 1 2 3 4 ... 17 18 19 20
    Data variables:
        obs:y1                   (realization, forecast_reference_time, lead_time, station) float64 363kB ...


## Build the README

```
poetry run jupyter nbconvert --execute --to markdown README.ipynb
```
