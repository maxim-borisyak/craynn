import theano.tensor as T

from ..layers import *
from ..layers.conv_ops import get_conv_nonlinearity

__all__ = [
  'maxout2d',
  'maxdropout2d'
]

def _maxout(incoming, conv_op, pool_size=5, f=None):
  f = get_conv_nonlinearity(f)
  return nonlinearity(f)(
    feature_pool(pool_size, f=T.max)(conv_op(incoming))
  )

def _maxdropout(incoming, conv_op, pool_size=5, p=None, f=None):
  f = get_conv_nonlinearity(f)
  return nonlinearity(f)(
    feature_pool(pool_size, f=T.max)(
      dropout(p=p, rescale=False)(
        conv_op(incoming)
      )
    )
  )


maxout2d = lambda num_filters, pool_size=4, f=None, conv=conv: lambda incoming:\
  _maxout(
    incoming=incoming,
    conv_op=conv(num_filters=num_filters * pool_size, f=lambda x: x),
    pool_size=pool_size,
    f=f
  )

maxdropout2d = lambda num_filters, pool_size=4, p=0.2, f=None, conv=conv: lambda incoming: \
  _maxdropout(
    incoming=incoming,
    conv_op=conv(num_filters=num_filters * pool_size, f=lambda x: x),
    pool_size=pool_size,
    p=p,
    f=f
  )