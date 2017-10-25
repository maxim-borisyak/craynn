import theano.tensor as T
from lasagne import *

from ..layers.conv_ops import get_companion_nonlinearity

__all__ = [
  'companion', 'max_companion', 'mean_companion'
]

def _conv_companion(layer, pool_function=T.max, n_units=None, nonlinearity=None):
  net = layers.GlobalPoolLayer(layer, pool_function=pool_function)

  if n_units is None:
    nonlinearity = get_companion_nonlinearity(n_units, nonlinearity)
    net = layers.DenseLayer(net, num_units=1, nonlinearity=nonlinearity)
    net = layers.FlattenLayer(net, outdim=1)
  else:
    nonlinearity = get_companion_nonlinearity(n_units, nonlinearity)
    net = layers.DenseLayer(net, num_units=n_units, nonlinearity=nonlinearity)

  return net

companion = lambda n_units=None, pool=T.max, f=None: lambda incoming: \
  _conv_companion(incoming, pool_function=pool, n_units=n_units, nonlinearity=f)

max_companion = lambda n_units=None, f=None: lambda incoming: \
  _conv_companion(incoming, n_units=n_units, pool_function=T.max, nonlinearity=f)

mean_companion = lambda n_units=None, f=None: lambda incoming: \
  _conv_companion(incoming, n_units=n_units, pool_function=T.mean, nonlinearity=f)