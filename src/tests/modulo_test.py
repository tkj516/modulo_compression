import sys
sys.path.append('../..')

# After setting up the path import modules
import tensorflow as tf
from src.modules.modulo import ModuloQuantize
import unittest


class TestModuloQuantize(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.inputs = tf.Variable(tf.random.uniform((8, 8, 128), -50., 50.))
        self.model = ModuloQuantize(3, -1, 0)

    def test_gpus(self):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        self.assertEqual(
            2, len(tf.config.list_physical_devices('GPU')), 
            "Not able to see all the GPUs")

    def test_train_step(self):
        with tf.GradientTape() as tape:
            outs = self.model(self.inputs)
        self.assertEqual(outs.shape, self.inputs.shape, "Shapes are not the same.")
        grads = tape.gradient(outs, self.inputs)

        self.model.summary()


if __name__ == "__main__":
    unittest.main()
