from keras.src.layers import Layer
from typing import Optional, Union, Literal
import keras
import keras.ops as ops
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
                skip_connection: bool = False,
                indx=0):
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
            self.layers.append(Dense(units, name=f"dense_{indx}:{i}"))
            if batchnorm:
                self.layers.append(BatchNormalization())
            self.layers.append(Activation(activations[i]))
            if i < len(dropout) and 0.0 < dropout[i] < 1.0:
                if mc_dropout:
                    self.layers.append(MonteCarloDropout(dropout[i], name=f"mc_dropout_{indx}:{i}"))
                else:
                    self.layers.append(Dropout(dropout[i], name=f"dropout_{indx}:{i}"))
            
        if skip_connection:
            self.skip_enc = Dense(hidden_layers[-1], name=f"skip_dense")
            self.skip_add = Add(name=f"skip_add")
            self.skip_act = Activation(activation=activations[-1], name=f"skip_activation")
            
            
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.hidden_layers[-1])
        
    
    def call(self, inputs):
        # iterate layers
        out = inputs
        for l in self.layers:
            out = l(out)
        # optional skip connection
        if self.skip_conn:
            inputs = self.skip_enc(inputs)
            out = self.skip_add([out, inputs])
            out = self.skip_act(out)
        return out
    
    
    
class MultibranchLayer(Layer):
    def __init__(self, branches: list[Layer], aggregation:  Literal['sum', 'concat']='concat'):
        super().__init__()
        
        self.branches = branches
        self.aggr = keras.layers.Concatenate(axis=1) if aggregation == 'concat' else keras.layers.Add()
        
        
    def call(self, inputs):
        branch_outputs = [branch(inputs) for branch in self.branches]
        return self.aggr(branch_outputs)
    
class CrossNetLayer(keras.layers.Layer):
    def __init__(self, hidden_size, depth=1):
        super().__init__()
        
        self.ws = [
            self.add_weight(
            shape=(hidden_size, 1),
            initializer="random_normal",
            trainable=True)
            for _ in range(depth)
        ]
        self.bs = [
            self.add_weight(
            shape=(hidden_size, 1),
            initializer="random_normal",
            trainable=True)
            for _ in range(depth)
        ]
        
        self.hidden_size = hidden_size
        self.encoder = Dense(self.hidden_size)
        
    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, x):
        x = self.encoder(x)
        # x_{l+1} = x_0*x_l^T*w_l + b_l + x_l = f(x_l, w_l, b_l) + x_l
        # where f learns the residual of x_{l+1} - x+l
        x0 = x
        x_l = x
        for l in range(len(self.ws)):
            outer_prod = x0.unsqueeze(2) * x_l.unsqueeze(1) 
            residual = ops.matmul(outer_prod, self.ws[l]) + self.bs[l] 
            residual = residual.squeeze()

            x_l = residual + x_l 
            
        return x_l
            
    def compute_output_shape(self, input_shape, *args, **kwargs):
        return (input_shape[0], self.hidden_size)
                        
                        
class ParallelConcatenateLayer(Layer):
    """Feeds the same input to all given layers 
    and concatenates their outputs along the last dimension.
    """
    def __init__(self, layers: list[Layer]):
        super().__init__()
        
        self.layers = layers
        
    def call(self, inputs):
        
        return keras.layers.Concatenate(axis=-1)([l(inputs) for l in self.layers])