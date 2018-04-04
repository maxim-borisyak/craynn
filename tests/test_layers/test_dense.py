import numpy as np

import tensorflow as tf

from craynn.layers import *


def test_dense():
  config = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU': 0},
    allow_soft_placement=True,
    log_device_placement=False
  )

  X = tf.placeholder(dtype='float32', shape=(None, 28), name='X')

  input_layer = InputLayer(shape=(None, 28), variable=X, )
  dense1 = DenseLayer(input_layer, num_units=13, name='dense1')
  dense2 = DenseLayer(dense1, num_units=3, name='dense1')

  assert len(get_all_params(dense2)) == 4
  assert len(get_all_params(dense2, trainable=True)) == 4
  assert len(get_all_params(dense2, trainable=False)) == 0
  assert len(get_all_params(dense2, weights=False)) == 2
  assert len(get_all_params(dense2, biases=True)) == 2
  assert len(get_all_params(dense2, blabla=True)) == 0

  features = get_output([dense2])

  with tf.Session(config=config) as s:
    s.run(tf.global_variables_initializer())

    result, = s.run(features, {
      X : np.random.uniform(size=(101, 28)).astype('float32')
    })

    assert result.shape == (101, 3)
    assert result.dtype == 'float32'
