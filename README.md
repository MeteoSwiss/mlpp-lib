# mlpp-lib

[![.github/workflows/run-tests.yml](https://github.com/MeteoSwiss/mlpp-lib/actions/workflows/run-tests.yml/badge.svg)](https://github.com/MeteoSwiss/mlpp-lib/actions/workflows/run-tests.yml)
[![pypi](https://img.shields.io/pypi/v/mlpp-lib.svg?colorB=<brightgreen>)](https://pypi.python.org/pypi/mlpp-lib/)

Collection of methods for ML-based postprocessing of weather forecasts.

:warning: **The code in this repository is currently work-in-progress and not recommended for production use.** :warning:



# Quickstart


```python
import numpy as np 
import xarray as xr 
import pandas as pd

from mlpp_lib.datasets import DataModule, DataSplitter
```

    2024-03-12 11:01:48.532698: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-12 11:01:48.594233: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-12 11:01:48.595154: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


    2024-03-12 11:01:49.442240: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



```python
LEADTIMES = np.arange(24)
REFTIMES = pd.date_range("2018-01-01", "2018-03-31", freq="24H")
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

    <xarray.Dataset>
    Dimensions:                  (forecast_reference_time: 90, lead_time: 24,
                                  station: 25)
    Coordinates:
      * forecast_reference_time  (forecast_reference_time) datetime64[ns] 2018-01...
      * lead_time                (lead_time) int64 0 1 2 3 4 5 ... 18 19 20 21 22 23
      * station                  (station) <U3 'AAA' 'BBB' 'CCC' ... 'XXX' 'YYY'
    Data variables:
        coe:x1                   (forecast_reference_time, lead_time, station) float64 ...
        coe:x2                   (forecast_reference_time, lead_time, station) float64 ...
        obs:x3                   (forecast_reference_time, lead_time, station) float64 ...
        dem:x4                   (forecast_reference_time, lead_time, station) float64 ...



```python
targets = targets_dataset()
print(targets)
```

    <xarray.Dataset>
    Dimensions:                  (forecast_reference_time: 90, lead_time: 24,
                                  station: 25)
    Coordinates:
      * forecast_reference_time  (forecast_reference_time) datetime64[ns] 2018-01...
      * lead_time                (lead_time) int64 0 1 2 3 4 5 ... 18 19 20 21 22 23
      * station                  (station) <U3 'AAA' 'BBB' 'CCC' ... 'XXX' 'YYY'
    Data variables:
        obs:y1                   (forecast_reference_time, lead_time, station) float64 ...
        obs:y2                   (forecast_reference_time, lead_time, station) float64 ...


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

## Training
The library builds on top of the tensorflow + keras API and provides some useful methods to quickly build probabilistic models, as well as a collection of probabilistic metrics. Of course, you're free to use tensorflow and tensorflow probability to build your own custom model. MLPP won't get in your way!


```python
from mlpp_lib.models import fully_connected_network
from mlpp_lib.losses import crps_energy
import tensorflow as tf 

model: tf.keras.Model = fully_connected_network(
    input_shape = datamodule.train.x.shape[1:],
    output_size = datamodule.train.y.shape[-1],
    hidden_layers = [32, 32],
    activations = "relu",
    probabilistic_layer = "IndependentNormal"
)

model.compile(loss=crps_energy, optimizer="adam")

history = model.fit(
    datamodule.train.x, datamodule.train.y,
    epochs = 2,
    batch_size = 32,
    validation_data = (datamodule.val.x, datamodule.val.y)
)
```

    Epoch 1/2
    689/689 [==============================] - 2s 2ms/step - loss: 0.5633 - val_loss: 0.5721

    Epoch 2/2
    689/689 [==============================] - 1s 2ms/step - loss: 0.5607 - val_loss: 0.5695


## Predictions
Once your model is trained, you can make predictions and create ensembles by sampling from the predictive distribution. The `Dataset` class comes with a method to wrap your ensemble predictions in a xarray object with the correct dimensions and coordinates.


```python
test_pred_ensemble = model(datamodule.test.x).sample(21)
test_pred_ensemble = datamodule.test.dataset_from_predictions(test_pred_ensemble, ensemble_axis=0)
print(test_pred_ensemble)
```

    <xarray.Dataset>
    Dimensions:                  (realization: 21, forecast_reference_time: 18,
                                  lead_time: 24, station: 5)
    Coordinates:
      * forecast_reference_time  (forecast_reference_time) datetime64[ns] 2018-03...
      * lead_time                (lead_time) int64 0 1 2 3 4 5 ... 18 19 20 21 22 23
      * station                  (station) <U3 'AAA' 'EEE' 'JJJ' 'PPP' 'RRR'
      * realization              (realization) int64 0 1 2 3 4 5 ... 16 17 18 19 20
    Data variables:
        obs:y1                   (realization, forecast_reference_time, lead_time, station) float64 ...
