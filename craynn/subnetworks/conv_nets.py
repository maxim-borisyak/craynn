import theano.tensor as T

from ..layers import *
from ..layers.conv_ops import get_companion_nonlinearity

from .common import repeat

__all__ = [
  'conv_block', 'double_conv',
  'diff_block', 'double_diff',
  'companion', 'max_companion', 'mean_companion'
]

conv_block = lambda n, num_filters, f=None: repeat(n)(
  conv(num_filters, f)
)

double_conv = lambda num_filters, f=None: conv_block(2, num_filters, f)

def _conv_companion(incoming, num_units=None, global_pool_op=global_pool(T.max), dense=dense, f=None):
  net = global_pool_op(incoming)

  if num_units is None:
    f = get_companion_nonlinearity(num_units, f)
    net = dense(num_units=1, f=f)(net)
    net = flatten(outdim=1)(net)
  else:
    f = get_companion_nonlinearity(num_units, f)
    net = dense(num_units=num_units, f=f)(net)

  return net

companion = lambda num_units=None, global_pool_op=global_pool(T.max), f=None, dense=dense: lambda incoming: \
  _conv_companion(incoming, global_pool_op=global_pool_op, num_units=num_units, f=f, dense=dense)

max_companion = lambda num_units=None, global_pool_op=global_pool(T.max), f=None, dense=dense: lambda incoming: \
  _conv_companion(incoming, global_pool_op=global_pool_op, num_units=num_units, f=f, dense=dense)

mean_companion = lambda num_units=None, global_pool_op=global_pool(T.mean), f=None, dense=dense: lambda incoming: \
  _conv_companion(incoming, global_pool_op=global_pool_op, num_units=num_units, f=f, dense=dense)

diff_block = lambda n, num_filters, f=None: repeat(n)(
  diff(num_filters, f)
)

double_diff = lambda num_filters, f=None: diff_block(2, num_filters, f)