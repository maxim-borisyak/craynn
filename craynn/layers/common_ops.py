import tensorflow as tf

from .layers_meta import FunctionalLayer
from .nonlinearities import leaky_relu, linear

__all__ = [
  'take', 'select', 'nothing',
  'get_nonlinearity',
  'default_common_nonlinearity',

  'NonlinearityLayer', 'nonlinearity',
  'ReshapeLayer', 'reshape'
]


default_common_nonlinearity = leaky_relu(0.05)
get_nonlinearity = lambda f=None: linear() if f is None else f

nothing = lambda incoming: incoming

class Select(object):
  def __getitem__(self, item):
    return lambda incomings: incomings[item]

select = Select()
take = select

class NonlinearityLayer(FunctionalLayer):
  def __init__(self, incoming, f=default_common_nonlinearity, name=None):
    super(NonlinearityLayer, self).__init__(incoming, name=name)

    self.f = f

  def get_output_for(self, incoming):
    return self.f(incoming)

  def get_output_shape_for(self, input_shape):
    return input_shape

nonlinearity = lambda f=None, name=None: lambda incoming: \
  NonlinearityLayer(incoming, f=get_nonlinearity(f), name=name)

class ReshapeLayer(FunctionalLayer):
  def __init__(self, incoming, shape, name=None):
    super(ReshapeLayer, self).__init__(incoming, name=name)
    self.shape = shape

  def get_output_for(self, incoming):
    return tf.reshape(incoming, shape=self.shape)

  def get_output_shape_for(self, input_shape):
    return tuple(None if s == -1 else s for s in self.shape)

reshape = lambda *shape, name=None: lambda incoming:\
  ReshapeLayer(incoming, shape)
