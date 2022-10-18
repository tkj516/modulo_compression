from typing import Any, Callable, Tuple
import tensorflow as tf


class Round(tf.keras.layers.Layer):
    def __init__(self):
        super(Round, self).__init__()

    @tf.custom_gradient
    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, Callable]:
        def grad(upstream):
            return upstream
        return tf.floor(inputs), grad


class Modulo(tf.keras.layers.Layer):
    def __init__(self, num_bits: int):
        super(Modulo, self).__init__()
        self.num_bits = tf.constant(num_bits)

    @tf.custom_gradient
    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, Callable]:
        def grad(upstream):
            return upstream

        num_bins = tf.cast(2 ** self.num_bits, inputs.dtype)
        rem = inputs - tf.math.floordiv(inputs, num_bins) * num_bins
        return rem, grad


class AdditiveUniformNoise(tf.keras.layers.Layer):
    def __init__(self, min_value: float = -1.0, max_value: float = 0.0):
        super(AdditiveUniformNoise, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        u = tf.random.uniform(
            tf.shape(inputs),
            minval=self.min_value,
            maxval=self.max_value,
            dtype=inputs.dtype
        )
        perturbed = inputs + u

        return perturbed, u


class ModuloQuantize(tf.keras.layers.Layer):
    def __init__(self, num_bits: int, min_value: float, max_value: float):
        super(ModuloQuantize, self).__init__()

        self.round = Round()
        self.modulo = Modulo(num_bits)
        self.additive_uniform_noise = AdditiveUniformNoise(
            min_value, max_value)

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        perturbed, u = self.additive_uniform_noise(inputs)

        if training:
            return self.modulo(perturbed)
        else:
            return self.modulo(self.modulo(self.round) - u)
