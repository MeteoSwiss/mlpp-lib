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
        self.activations = activations
            
        if isinstance(dropout, list):
            assert len(dropout) == len(hidden_layers)
        elif isinstance(dropout, float):
            self.dropout = [dropout] * (len(hidden_layers))
        else:
            self.dropout = []

        
        self.skip_conn = skip_connection
        self.layers = []
        self.hidden_layers = hidden_layers
        self.batchnorm = batchnorm
        self.mc_dropout = mc_dropout

    
    def build(self, input_shape):
        
        for i,units in enumerate(self.hidden_layers):
            self.layers.append(Dense(units, name=f"dense_{i}"))
            if self.batchnorm:
                self.layers.append(BatchNormalization())
            self.layers.append(Activation(self.activations[i]))
            if i < len(self.dropout) and 0.0 < self.dropout[i] < 1.0:
                if self.mc_dropout:
                    self.layers.append(MonteCarloDropout(self.dropout[i], name=f"mc_dropout_{i}"))
                else:
                    self.layers.append(Dropout(self.dropout[i], name=f"dropout_{i}"))
            
        if self.skip_conn:
            self.skip_enc = Dense(input_shape[-1], name=f"skip_dense")
            self.skip_add = Add(name=f"skip_add")
            self.skip_act = Activation(activation=self.activations[-1], name=f"skip_activation")
        
    
    def call(self, inputs):
        # iterate layers
        out = inputs
        for l in self.layers:
            out = l(out)
        # optional skip connection
        if self.skip_conn:
            out = self.skip_enc(out)
            out = self.skip_add([out, inputs])
            out = self.skip_act(out)
        return out
    