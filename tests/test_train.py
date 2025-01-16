import torch
import scoringrules as sr
import keras 
import pytest
import json

import cloudpickle
import numpy as np
import pytest
from keras import Model
import xarray as xr

from mlpp_lib import train
from mlpp_lib.normalizers import DataTransformer
from mlpp_lib.datasets import DataModule, DataSplitter
from mlpp_lib.layers import MultilayerPerceptron
from mlpp_lib.losses import DistributionLossWrapper, SampleLossWrapper
from mlpp_lib.models import ProbabilisticModel
from mlpp_lib.probabilistic_layers import DistributionLayer, UniveriateGaussianModule

from .test_model_selection import ValidDataSplitterOptions


@pytest.mark.parametrize("loss_type", ['analytical', 'samples'], ids=['train=CRPS closed form', 'train=CRPS MC estimate'])
def test_train_noisy_polynomial(loss_type):
    # test whether the model can learn y = x^2 + e ~ N(0,sigma)
    num_samples = 1000
    x_values = torch.linspace(-1, 1, num_samples).reshape(-1,1)

    true_mean = x_values**2  # Mean centered at x^2 for x in [-1,1]
    true_std = 0.05 

    # Generate the dataset in torch
    y_values = torch.normal(mean=true_mean, std=true_std * torch.ones_like(true_mean)).reshape(-1,1)
    
    if loss_type == 'analytical':
        crps_normal = DistributionLossWrapper(fn=sr.crps_normal)
    else:
        crps_normal = SampleLossWrapper(fn=sr.crps_ensemble, num_samples=100)
    
    prob_layer = DistributionLayer(distribution=UniveriateGaussianModule(), num_samples=21)
    encoder = MultilayerPerceptron(hidden_layers=[16,8], 
                                  batchnorm=False, 
                                  skip_connection=False,
                                  activations='sigmoid')
    
    model = ProbabilisticModel(encoder=encoder, output_distribution=prob_layer)
    
    model(x_values[:100]) # infer shapes
    model.compile(loss=crps_normal, optimizer=keras.optimizers.Adam(learning_rate=0.1))
    
    history = model.fit(x=x_values, y=y_values, epochs=50, batch_size=200)
    
    # Assert it learned something
    assert history.history['loss'][-1] < 0.05
    


RUNS = [
    # minimal set of parameters using scoringrules
    {
        "features": ["coe:x1"],
        "targets": ["obs:y1"],
        "normalizer": {"default": "MinMaxScaler"},
        "model": {
            "fully_connected_network": {
                "hidden_layers": [10],
                "probabilistic_layer": "Normal",
            }
        },
        'loss': {
            'DistributionLossWrapper':
                'scoringrules.crps_normal'
                    # {'scoringrules.crps_normal': 
                    #     {
                    #         'num_samples': 100,
                    #     }
                    # }
        },
        "optimizer": "RMSprop",
        "callbacks": [
            {"EarlyStopping": {"patience": 10, "restore_best_weights": True}}
        ],
    },
    # minimal set of parameters using named loss from mlpp
    {
        "features": ["coe:x1"],
        "targets": ["obs:y1"],
        "normalizer": {"default": "MinMaxScaler"},
        "model": {
            "fully_connected_network": {
                "hidden_layers": [10],
                "probabilistic_layer": "Normal",
            }
        },
        'loss': {
            'CRPSEnsemble':
                {
                    'num_samples': 100,
                }
        },
        "optimizer": "RMSprop",
        "callbacks": [
            {"EarlyStopping": {"patience": 10, "restore_best_weights": True}}
        ],
    },
    # use MAEBusts
    {
        "features": ["coe:x1"],
        "targets": ["obs:y1"],
        "normalizer": {"default": "MinMaxScaler"},
        "model": {
            "fully_connected_network": {
                "hidden_layers": [10],
                "probabilistic_layer": "IndependentBeta",
            }
        },
        'loss': {
            'CRPSEnsemble':
                {
                    'num_samples': 100,
                }
        },
        "optimizer": {"Adam": {"learning_rate": 0.1, "beta_1": 0.95}},
        "metrics": ["expected_bias", "expected_mean_absolute_error", {"MAEBusts": {"threshold": 0.5}}],
    },
    # use a learning rate scheduler
    {
        "features": ["coe:x1"],
        "targets": ["obs:y1"],
        "normalizer": {"default": "MinMaxScaler"},
        "model": {
            "fully_connected_network": {
                "hidden_layers": [10],
                "probabilistic_layer": "Normal",
            }
        },
        "loss": "CRPSNormal",
        "optimizer": {
            "Adam": {
                "learning_rate": {
                    "CosineDecayRestarts": {
                        "initial_learning_rate": 0.001,
                        "first_decay_steps": 20,
                        "t_mul": 1.5,
                        "m_mul": 1.1,
                        "alpha": 0,
                    }
                }
            }
        },
        "callbacks": [
            {"EarlyStopping": {"patience": 10, "restore_best_weights": True}}
        ],
    },
    #
    {
        "features": ["coe:x1"],
        "targets": ["obs:y1"],
        "normalizer": {"default": "MinMaxScaler"},
        "model": {
            "fully_connected_network": {
                "hidden_layers": [10],
                "probabilistic_layer": "Normal",
                "skip_connection": True,
            }
        },
        'loss': {
            'CRPSEnsemble':
                {
                    'num_samples': 100,
                }
        },
        "metrics": ["expected_bias"],
        "callbacks": [
            {
                "EarlyStopping": {
                    "patience": 10,
                    "restore_best_weights": True,
                    "verbose": 1,
                }
            },
            {"ReduceLROnPlateau": {"patience": 1, "verbose": 1}},
            {"EnsembleMetrics": {"thresholds": [0, 1, 2]}},
        ],
    },
#     # with multiscale CRPS loss
#     {
#         "features": ["coe:x1"],
#         "targets": ["obs:y1"],
#         "normalizer": {"default": "MinMaxScaler"},
#         "model": {
#             "fully_connected_network": {
#                 "hidden_layers": [10],
#                 "probabilistic_layer": "IndependentNormal",
#             }
#         },
#         "group_samples": {"t": 2},
#         "loss": {
#             "MultiScaleCRPSEnergy": {"scales": [1, 2], "threshold": 0, "n_samples": 5}
#         },
#         "metrics": ["bias"],
#     },
#     # with combined loss
#     {
#         "features": ["coe:x1"],
#         "targets": ["obs:y1"],
#         "normalizer": {"default": "MinMaxScaler"},
#         "model": {
#             "fully_connected_network": {
#                 "hidden_layers": [10],
#                 "probabilistic_layer": "IndependentNormal",
#             }
#         },
#         "loss": {
#             "CombinedLoss": {
#                 "losses": [
#                     {"BinaryClassifierLoss": {"threshold": 1}, "weight": 0.7},
#                     {"WeightedCRPSEnergy": {"threshold": 0.1}, "weight": 0.1},
#                 ],
#             }
#         },
#     },
]


@pytest.fixture  # https://docs.pytest.org/en/6.2.x/tmpdir.html
def write_datasets_zarr(tmp_path, features_dataset, targets_dataset):
    features_dataset.to_zarr(tmp_path / "features.zarr", mode="w")
    targets_dataset.to_zarr(tmp_path / "targets.zarr", mode="w")


@pytest.mark.skipif("zarr" not in xr.backends.list_engines(), reason="missing zarr")
@pytest.mark.usefixtures("write_datasets_zarr")
@pytest.mark.parametrize("cfg", RUNS)
def test_train_fromfile(tmp_path, cfg):
    num_epochs = 3
    cfg.update({"epochs": num_epochs})

    splitter_options = ValidDataSplitterOptions(time="lists", station="lists")
    datasplitter = DataSplitter(
        splitter_options.time_split, splitter_options.station_split
    )
    datanormalizer = DataTransformer(**cfg["normalizer"])
    batch_dims = ["forecast_reference_time", "t", "station"]
    datamodule = DataModule(
        features=cfg["features"],
        targets=cfg["targets"],
        batch_dims=batch_dims,
        splitter=datasplitter,
        normalizer=datanormalizer,
        data_dir=tmp_path.as_posix() + "/",
    )
    results = train.train(cfg, datamodule)

    assert len(results) == 4
    assert isinstance(results[0], Model)  # model
    assert isinstance(results[1], dict)  # custom_objects
    assert isinstance(results[2], DataTransformer)  # normalizer
    assert isinstance(results[3], dict)  # history

    assert all([np.isfinite(v).all() for v in results[3].values()])
    assert all([len(v) == num_epochs for v in results[3].values()])

    # try to pickle the custom objects
    cloudpickle.dumps(results[1])

    # try to dump fit history to json
    json.dumps(results[3])


@pytest.mark.parametrize("cfg", RUNS)
def test_train_fromds(features_dataset, targets_dataset, cfg):
    num_epochs = 3
    cfg.update({"epochs": num_epochs})

    splitter_options = ValidDataSplitterOptions(time="lists", station="lists")
    datasplitter = DataSplitter(
        splitter_options.time_split, splitter_options.station_split
    )
    datanormalizer = DataTransformer(**cfg["normalizer"])
    batch_dims = ["forecast_reference_time", "t", "station"]
    datamodule = DataModule(
        features_dataset[cfg["features"]],
        targets_dataset[cfg["targets"]],
        batch_dims,
        splitter=datasplitter,
        normalizer=datanormalizer,
        group_samples=cfg.get("group_samples"),
    )
    results = train.train(cfg, datamodule)

    assert len(results) == 4
    assert isinstance(results[0], Model)  # model
    assert isinstance(results[1], dict)  # custom_objects
    assert isinstance(results[2], DataTransformer)  # normalizer
    assert isinstance(results[3], dict)  # history

    assert all([np.isfinite(v).all() for v in results[3].values()])
    assert all([len(v) == num_epochs for v in results[3].values()])

    # try to pickle the custom objects
    cloudpickle.dumps(results[1])

    # try to dump fit history to json
    json.dumps(results[3])
