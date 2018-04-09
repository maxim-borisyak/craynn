import numpy as np

import tensorflow as tf

from craynn.layers import *
from craynn.updates import *
from craynn.layers.nonlinearities import *

def test_sgd():
  config = tf.ConfigProto(
    device_count={'CPU': 1, 'GPU': 0},
    allow_soft_placement=True,
    log_device_placement=False
  )

  input_layer = InputLayer(shape=(None, 2))
  dense1 = DenseLayer(input_layer, nonlinearity=leaky_relu(0.05), num_units=3)
  dense2 = DenseLayer(dense1, nonlinearity=None, num_units=1)

  X = tf.placeholder(shape=(None, 2), dtype='float32')
  y = tf.placeholder(shape=(None, ), dtype='float32')

  p = get_output(dense2, substitutes={ input_layer : X })
  p = tf.reshape(p, shape=(-1, ))

  loss = tf.reduce_mean(
    y * tf.nn.softplus(-p) + (1 - y) * tf.nn.softplus(p)
  )

  params = get_all_params(dense2, trainable=True)
  upd = sgd(loss, params, learning_rate=1.0e-2)

  X_train = np.vstack([
    np.random.uniform(-1, 1, size=(32, 2)).astype('float32'),
    np.random.uniform(-0.25, 1.75, size=(32, 2)).astype('float32')
  ])

  y_train = np.hstack([
    np.ones(32, dtype='float32'),
    np.zeros(32, dtype='float32')
  ])

  l = np.log(2)

  with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())

    for i in range(1024):
      _, l = session.run([upd, loss], {
        X: X_train,
        y : y_train
      })

  assert l < np.log(2), '(S)GD is not optimizing!'