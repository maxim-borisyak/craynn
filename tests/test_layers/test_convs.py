import numpy as np

import tensorflow as tf

from craynn.layers import *

def check_kernel_layer(layer, session, input_shape, expected_shape=None):
  X = tf.placeholder(dtype='float32', shape=input_shape, name='X')

  net = InputLayer(shape=input_shape, variable=X, )
  net = layer(net)

  session.run(tf.global_variables_initializer())

  out_shape = get_output_shape(net)
  if expected_shape is not None:
    assert expected_shape == out_shape

  output = get_output(net)

  n = np.random.permutation(np.arange(32))[0]

  result = session.run(output, {
    X: np.random.uniform(size=(n,) + input_shape[1:]).astype('float32')
  })
  assert result.shape == (n, ) + out_shape[1:]

def test_2D_convs():
  config = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU': 0},
    allow_soft_placement=True,
    log_device_placement=False
  )
  s = tf.Session(config=config)

  check = lambda layer, expected_shape=None: check_kernel_layer(layer, s, (None, 35, 35, 3), expected_shape)

  for pad in ('valid', 'same'):
    for num_filters in (7, 2, 3):
      for stride in (4, 1, (2, 2)):
        for filter_size in (1, 2, (3, 3), (7, 7)):
          check(
            conv2d(num_filters=num_filters, filter_size=filter_size, stride=stride, pad=pad)
          )

  check(
    conv2d(num_filters=7, filter_size=3, stride=(2, 3), pad='valid'),
    expected_shape=(None, 17, 11, 7)
  )

def test_conv_deconv():
  config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False
  )
  s = tf.Session(config=config)

  X = tf.placeholder(dtype='float32', shape=(None, 28, 28, 3), name='X')

  net = InputLayer(shape=(None, 28, 28, 3), variable=X, )

  net = Conv2DLayer(net, num_filters=12)
  net = TransposedConv2DLayer(net, num_filters=3)

  s.run(tf.global_variables_initializer())

  X_ = get_output(net)

  input = np.ones(shape=(13, 28, 28, 3), dtype='float32')

  result = s.run(X_, {
    X : input
  })

  assert result.shape == input.shape