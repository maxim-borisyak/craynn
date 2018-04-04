from craynn.viz.visualize import remove_craynn_layers_prefix
from craynn.layers import *

from craynn.viz import *

def test_naming():
  assert remove_craynn_layers_prefix(
    ['craynn.layers.dense.DenseLayer', 'craynn.layers.dense.DenseLayer2']
  ) == ['DenseLayer', 'DenseLayer2']

  assert remove_craynn_layers_prefix(
    ['craynn.layers.dense.DenseLayer', 'craynn.layers.dense.DenseLayer2', 'craynn.layers.dense.restricted.DenseLayer']
  ) == ['DenseLayer', 'DenseLayer2', 'restricted.DenseLayer']

  assert remove_craynn_layers_prefix(
    ['craynn.layers.dense.DenseLayer', 'craynn.layers.dense.DenseLayer2', 'keras.layers.dense.DenseLayer']
  ) == ['DenseLayer', 'DenseLayer2', 'keras.layers.dense.DenseLayer']

  assert remove_craynn_layers_prefix(
    ['craynn.layers.dense.DenseLayer', 'craynn.layers.dense.DenseLayer2', 'DenseLayer']
  ) == ['craynn.layers.dense.DenseLayer', 'craynn.layers.dense.DenseLayer2', 'DenseLayer']

def test_draw():
  import tensorflow as tf

  config = tf.ConfigProto(
    device_count={'CPU': 1, 'GPU': 0},
    allow_soft_placement=True,
    log_device_placement=False
  )

  X = tf.placeholder(dtype='float32', shape=(None, 28), name='X')

  input_layer = InputLayer(shape=(None, 28), variable=X, )
  dense1 = DenseLayer(input_layer, num_units=13)
  dense2 = DenseLayer(dense1, num_units=3, name='dense1')

  draw_to_file(dense2, 'test_network.png', params=('Params:', viz_params(weights=None)), parameters=('parameters', viz_all_params()))