# Code adapted from https://github.com/tensorflow/compression/blob/v2.10.0/models/ms2020.py

import tensorflow as tf
import tensorflow_compression as tfc
import functools


class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  def __init__(self, latent_depth):
    super().__init__()
    conv = functools.partial(tfc.SignalConv2D, corr=True, strides_down=2,
                             padding="same_zeros", use_bias=True)
    layers = [
        tf.keras.layers.Lambda(lambda x: x / 255.),
        conv(192, (5, 5), name="layer_0", activation=tfc.GDN(name="gdn_0")),
        conv(192, (5, 5), name="layer_1", activation=tfc.GDN(name="gdn_1")),
        conv(192, (5, 5), name="layer_2", activation=tfc.GDN(name="gdn_2")),
        conv(latent_depth, (5, 5), name="layer_3", activation=None),
    ]
    for layer in layers:
      self.add(layer)


class SynthesisTransform(tf.keras.Sequential):
  """The synthesis transform."""

  def __init__(self):
    super().__init__()
    conv = functools.partial(tfc.SignalConv2D, corr=False, strides_up=2,
                             padding="same_zeros", use_bias=True)
    layers = [
        conv(192, (5, 5), name="layer_0",
             activation=tfc.GDN(name="igdn_0", inverse=True)),
        conv(192, (5, 5), name="layer_1",
             activation=tfc.GDN(name="igdn_1", inverse=True)),
        conv(192, (5, 5), name="layer_2",
             activation=tfc.GDN(name="igdn_2", inverse=True)),
        conv(3, (5, 5), name="layer_3",
             activation=None),
        tf.keras.layers.Lambda(lambda x: x * 255.),
    ]
    for layer in layers:
      self.add(layer)