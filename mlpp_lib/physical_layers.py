import tensorflow as tf
from tensorflow.keras.layers import Layer


class ThermodynamicLayer(Layer):
    """
    Physical layer based on empirical approximations of thermodynamic
    state equations. The following equations were used:

    Vapor pressure: formula from Bolton (1980) for T in degrees Celsius:
    e = 6.112 * exp(17.67 * T / (T + 243.5))

    Mixing ratio:
    r = 622.0 * e / (p - e)

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # indices layer inputs
        self.T_idx = 0  # air_temperature
        self.D_idx = 1  # dew_point_deficit
        self.P_idx = 2  # surface_air_pressure

        self.EPSILON = tf.constant(622.0)

    def build(self, input_shape: tuple[int]) -> None:
        super().build(input_shape)

    def get_config(self) -> None:
        base_config = super().get_config()
        return dict(list(base_config.items()))

    def call(self, inputs):

        air_temperature = inputs[..., self.T_idx]
        dew_point_deficit = inputs[..., self.D_idx]
        surface_air_pressure = inputs[..., self.P_idx]

        dew_point_temperature = air_temperature - tf.nn.relu(dew_point_deficit)
        water_vapor_saturation_pressure = 6.112 * tf.exp(
            (17.67 * air_temperature) / (air_temperature + 243.5)
        )
        water_vapor_pressure = 6.112 * tf.exp(
            (17.67 * dew_point_temperature) / (dew_point_temperature + 243.5)
        )
        relative_humidity = (
            water_vapor_pressure / water_vapor_saturation_pressure * 100.0
        )
        humidity_mixing_ratio = self.EPSILON * (
            water_vapor_pressure / (surface_air_pressure - water_vapor_pressure)
        )

        out = tf.concat(
            [
                air_temperature[..., None],
                dew_point_temperature[..., None],
                surface_air_pressure[..., None],
                relative_humidity[..., None],
                humidity_mixing_ratio[..., None],
            ],
            axis=-1,
        )

        return out
