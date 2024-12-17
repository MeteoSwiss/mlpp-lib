import torch
import pytest

from mlpp_lib.probabilistic_layers import (
    BaseDistributionLayer, 
    MultivariateGaussianTriLModule,
    UnivariateCensoredGaussianModule
)
from mlpp_lib.probabilistic_layers import MissingReparameterizationError


def test_multivariate_gaussian():
    distr = MultivariateGaussianTriLModule(dim=4)
    multivariate_gaussian_layer = BaseDistributionLayer(distribution=distr, num_samples=21)

    inputs = torch.randn(16,8)
    
    # ensure you can sample, ie the generated matrix L is a valid Cholesky lower triangular
    multivariate_gaussian_layer(inputs, output_type='samples')
    
    
def test_defense_missing_rsample():
    # censored normal does not have rsample so far
    distr = UnivariateCensoredGaussianModule(a=-1., b=1.)
    censored_gaussian_layer = BaseDistributionLayer(distribution=distr, num_samples=21)
    # ensure that trying to call the layer in training mode requiring samples raises an error 
    with pytest.raises(MissingReparameterizationError):
        censored_gaussian_layer(torch.randn(32,4), output_type='samples', training=True)
    
    
# from inspect import getmembers, isclass

# import numpy as np
# import pytest
# import tensorflow as tf
# import tensorflow_probability as tfp

# from mlpp_lib import models
# from mlpp_lib import probabilistic_layers
# from mlpp_lib.datasets import Dataset


# LAYERS = [obj[0] for obj in getmembers(probabilistic_layers, isclass)]


# @pytest.mark.parametrize("layer", LAYERS)
# def test_probabilistic_layers(layer):

#     layer_class = getattr(probabilistic_layers, layer)

#     output_size = 2
#     params_size = layer_class.params_size(output_size)

#     # build
#     layer = layer_class(output_size)

#     # call
#     input_tensor = tf.random.normal((10, params_size))
#     output_dist = layer(input_tensor)

#     n_samples = 5
#     samples = output_dist.sample(n_samples)

#     assert samples.shape == (n_samples, 10, output_size)


# @pytest.mark.parametrize("layer", LAYERS)
# def test_probabilistic_model(layer):
#     """Test model compile with prob layers"""

#     layer_class = getattr(probabilistic_layers, layer)
#     tfkl = tf.keras.layers
#     input_shape = [28, 28, 1]
#     encoded_shape = 2
#     encoder = tf.keras.Sequential(
#         [
#             tfkl.InputLayer(input_shape=input_shape),
#             tfkl.Flatten(),
#             tfkl.Dense(10, activation="relu"),
#             tfkl.Dense(layer_class.params_size(encoded_shape)),
#             layer_class(encoded_shape),
#         ]
#     )
#     encoder.summary()
#     encoder.compile()
#     assert isinstance(encoder, tf.keras.Sequential)
#     model_output = encoder.layers[-1].output
#     assert not isinstance(
#         model_output, list
#     ), "The model output must be a single tensor!"
#     assert (
#         len(model_output.shape) < 3
#     ), "The model output must be a vector or a single value!"


# @pytest.mark.parametrize("layer", LAYERS)
# def test_probabilistic_model_predict(layer, features_dataset, targets_dataset):
#     batch_dims = ["forecast_reference_time", "t", "station"]
#     data = (
#         Dataset.from_xarray_datasets(features_dataset, targets_dataset)
#         .stack(batch_dims)
#         .drop_nans()
#     )
#     x_shape = data.x.shape
#     y_shape = data.y.shape
#     input_shape = x_shape[1]
#     output_size = y_shape[1]
#     model = models.fully_connected_network(
#         input_shape, output_size, hidden_layers=[3], probabilistic_layer=layer
#     )
#     out_predict = model.predict(data.x)
#     assert isinstance(out_predict, np.ndarray)
#     assert out_predict.ndim == 2
#     assert out_predict.shape[0] == data.y.shape[0]
#     assert out_predict.shape[1] == data.y.shape[-1]
#     out_distr = model(data.x)
#     assert isinstance(out_distr, tfp.distributions.Distribution)
#     num_samples = 2
#     out_samples = out_distr.sample(num_samples)
#     assert isinstance(out_samples, tf.Tensor)
#     assert out_samples.ndim == 3
#     assert out_samples.shape[0] == num_samples
#     assert out_samples.shape[1] == data.y.shape[0]
#     assert out_samples.shape[2] == data.y.shape[-1]
