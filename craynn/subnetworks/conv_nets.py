import theano.tensor as T
from lasagne import *

from ..layers import *
from .common import complete_conv_kwargs, complete_deconv_kwargs, get_deconv_kwargs
from .common import chain

from ..layers.conv_ops import get_conv_nonlinearity, get_companion_nonlinearity

__all__ = [
  'cnn',
  'decnn',
  'cae',
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

def cnn(num_filters, conv_op=conv, pool_op=max_pool, companion=max_companion()):
  block = lambda n_filters: (conv_op(n), pool_op((2, 2)))

  blocks = [
    block(n) if (i < len(num_filters) - 1) else conv_op(n)
    for i, n in enumerate(num_filters)
  ] + [
    companion
  ]

  return chain(*blocks)

def decnn(input_layer, num_filters, **deconv_kwargs):
  net = input_layer
  deconv_kwargs = complete_deconv_kwargs(deconv_kwargs)

  for i, n_filters in enumerate(num_filters):
    net = layers.Upscale2DLayer(
      net, scale_factor=(2, 2),
      name='depool%d' % i,
    )

    net = layers.Deconv2DLayer(
      net,
      num_filters=n_filters,
      name = 'conv%d' % i,
      **deconv_kwargs
    )

  return net

def cae(input_layer, n_channels, **conv_kwargs):
  initial_channels = layers.get_output_shape(input_layer)[1]

  conv_kwargs = complete_conv_kwargs(conv_kwargs)
  deconv_kwargs = get_deconv_kwargs(conv_kwargs)

  net = input_layer
  net = cnn(net, n_channels, last_pool=True, **conv_kwargs)
  deconv_channels = ((initial_channels, ) + tuple(n_channels[:-1]))[::-1]
  net = decnn(net, deconv_channels, **deconv_kwargs)

  return net