from keras.src.layers import Layer
from typing import Optional, Union
import keras
from keras.src.layers import (
    Add,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
)

@keras.saving.register_keras_serializable()
class MonteCarloDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

@keras.saving.register_keras_serializable()
class FullyConnectedLayer(Layer):
    """ A fully connected layer composed of a sequence 
        of linear layers interleaved by optional 
        batch norms, and dropouts/MC dropouts.
    """
    def __init__(self,
                hidden_layers: list,
                batchnorm: bool = False,
                activations: Optional[Union[str, list[str]]] = "relu",
                dropout: Optional[Union[float, list[float]]] = None,
                mc_dropout: bool = False,
                skip_connection: bool = False,):
        super().__init__()
        
        if isinstance(activations, list):
            assert len(activations) == len(hidden_layers)
        elif isinstance(activations, str):
            activations = [activations] * len(hidden_layers)
            
        if isinstance(dropout, list):
            assert len(dropout) == len(hidden_layers)
        elif isinstance(dropout, float):
            dropout = [dropout] * (len(hidden_layers))
        else:
            dropout = []

        
        self.skip_conn = skip_connection
        self.layers = []
        self.hidden_layers = hidden_layers
        
        for i,units in enumerate(hidden_layers):
            self.layers.append(Dense(units, name=f"dense_{i}"))
            if batchnorm:
                self.layers.append(BatchNormalization())
            self.layers.append(Activation(activations[i]))
            if i < len(dropout) and 0.0 < dropout[i] < 1.0:
                if mc_dropout:
                    self.layers.append(MonteCarloDropout(dropout[i], name=f"mc_dropout_{i}"))
                else:
                    self.layers.append(Dropout(dropout[i], name=f"dropout_{i}"))
            
        if skip_connection:
            self.skip_enc = Dense(hidden_layers[-1], name=f"skip_dense")
            self.skip_add = Add(name=f"skip_add")
            self.skip_act = Activation(activation=activations[-1], name=f"skip_activation")
            
            
    def compute_output_shape(self, input_shape):
        if self.skip_conn:
            return input_shape
        return (input_shape[0], self.hidden_layers[-1])
        
    
    def call(self, inputs):
        # iterate layers
        out = inputs
        for l in self.layers:
            out = l(out)
        # optional skip connection
        if self.skip_conn:
            out = self.skip_enc(inputs)
            out = self.skip_add([out, inputs])
            out = self.skip_act(out)
        return out
    