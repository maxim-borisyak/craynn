import tensorflow as tf

from .meta import FunctionalLayer

from ..nonlinearities import leaky_relu, linear

__all__ = [
  'take', 'select', 'nothing',
  'get_common_nonlinearity',
  'nonlinearity'
]

get_common_nonlinearity = lambda f=None: (
  leaky_relu(0.05)
  if f == 'default' else
  (linear if f is None else f)
)

nothing = lambda incoming: incoming

class Select(object):
  def __getitem__(self, item):
    return lambda incomings: incomings[item]

select = Select()
take = select

class NonlinearityLayer(FunctionalLayer):
  def __init__(self, incoming, f=None, name=None):
    super(NonlinearityLayer, self).__init__(incoming, name=name)

    self.f = f

  def get_output_for(self, incoming):
    return self.f(incoming)

  def get_output_shape_for(self, input_shape):
    return input_shape

nonlinearity = lambda f=None, name=None: lambda incoming: \
  NonlinearityLayer(incoming, f=get_common_nonlinearity(f), name=name)
