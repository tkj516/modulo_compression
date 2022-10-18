import sys
sys.path.append('../..')

# After setting up the path import modules
import itertools
import unittest
import tensorflow as tf

from src.modules.nonlinear_transforms import AnalysisTransform, SynthesisTransform


class TestNonLinearTransform(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.inputs = tf.Variable(
            tf.random.uniform(
                (2, 256, 256, 3),
                minval=0,
                maxval=255,
                dtype=tf.dtypes.int64,
            )
        )
        self.analysis = AnalysisTransform(latent_depth=320)
        self.synthesis = SynthesisTransform()
        self.criterion = tf.keras.losses.MeanSquaredError()

    def test_gpus(self):
        print("Num GPUs Available: ", len(
            tf.config.list_physical_devices('GPU')))
        self.assertEqual(
            2, len(tf.config.list_physical_devices('GPU')),
            "Not able to see all the GPUs")

    def test_train_step(self):
        with tf.GradientTape() as tape:
            latents = self.analysis(tf.cast(self.inputs, tf.dtypes.float32))
            recons = self.synthesis(latents)
            loss = self.criterion(self.inputs, recons)
        self.assertEqual(recons.shape, self.inputs.shape,
                        "Shapes are not the same.")
        grads = tape.gradient(
                    loss, 
                    list(itertools.chain(
                        [
                            self.analysis.trainable_weights, 
                            self.synthesis.trainable_weights
                        ]
                    ))
                )


if __name__ == "__main__":
    unittest.main()
