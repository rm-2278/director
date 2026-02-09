import tensorflow as tf
import numpy as np

import sys
import os
sys.path.append(os.getcwd())

from embodied.agents.director import tfutils
from embodied.agents.director import nets

class TestModule(tfutils.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, x):
        return self.get('dense', tf.keras.layers.Dense, 10)(x)

model = TestModule()
x = tf.zeros((1, 5))
y = model(x)
print(f"Variables: {len(model.variables)}")
print(f"Trainable Variables: {len(model.trainable_variables)}")

if len(model.variables) == 0:
    print("WARNING: Variables NOT tracked in self._modules dict!")
else:
    print("VARIABLES TRACKED SUCCESSFULLY.")
