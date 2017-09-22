import numpy as np

import theano
import theano.tensor as T

from lasagne import *

from craynn import Expression

__all__ = [
  'Net', 'net',
  'factory',
  'get_input_layer',
  'get_noise_layer',
  'default_cls'
]

def factory(cls):
  return lambda *args, **kwargs: lambda input = None: cls(*args, input_layer=input, **kwargs)

def default_cls(cls):
  def gen(depth = 3, initial_filters = 4, *args, **kwargs):
    n_filters = [ initial_filters * (2**i) for i in range(depth) ]
    return cls(n_filters, *args, **kwargs)
  return gen

def get_input_layer(img_shape, input_layer):
  if input_layer is None:
    return layers.InputLayer(
      shape=(None,) + img_shape,
      name='input'
    )
  else:
    return input_layer

def get_noise_layer(input_layer, sigma=None):
  if sigma is not None:
    return layers.GaussianNoiseLayer(input_layer, sigma=sigma, name='noise')
  else:
    return input_layer


class Net(Expression):
  def __init__(self, factory, img_shape=None, input_layer=None):
    self.input_layer = get_input_layer(img_shape, input_layer)
    net = factory(self.input_layer)

    if not hasattr(net, '__iter__'):
      net = [net]

    super(Net, self).__init__([self.input_layer], net)

net = lambda factory, img_shape=None, input_layer=None: Net(factory, img_shape, input_layer)