import keras
from keras import ops
from keras.layers import Layer


@keras.saving.register_keras_serializable(package="mlpp_lib")
class ThermodynamicLayer(Layer):
    """
    Physical layer based on empirical approximations of thermodynamic
    state equations. The following equations were used:

    Vapor pressure: formula from Bolton (1980) for T in degrees Celsius:
    e = 6.112 * exp(17.67 * T / (T + 243.5))

    Mixing ratio:
    r = 622.0 * e / (p - e)

    """

    EPSILON = 622.0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # indices layer inputs
        self.T_idx = 0  # air_temperature
        self.D_idx = 1  # dew_point_deficit
        self.P_idx = 2  # surface_air_pressure

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], 5)

    def call(self, inputs):

        air_temperature = inputs[..., self.T_idx]
        dew_point_deficit = inputs[..., self.D_idx]
        surface_air_pressure = inputs[..., self.P_idx]

        dew_point_temperature = air_temperature - ops.relu(dew_point_deficit)
        water_vapor_saturation_pressure = 6.112 * ops.exp(
            (17.67 * air_temperature) / (air_temperature + 243.5)
        )
        water_vapor_pressure = 6.112 * ops.exp(
            (17.67 * dew_point_temperature) / (dew_point_temperature + 243.5)
        )
        relative_humidity = (
            water_vapor_pressure / water_vapor_saturation_pressure * 100.0
        )
        humidity_mixing_ratio = self.EPSILON * (
            water_vapor_pressure / (surface_air_pressure - water_vapor_pressure)
        )

        return ops.stack(
            [
                air_temperature,
                dew_point_temperature,
                surface_air_pressure,
                relative_humidity,
                humidity_mixing_ratio,
            ],
            axis=-1,
        )
