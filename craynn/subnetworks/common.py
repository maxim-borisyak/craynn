from lasagne import *
from ..layers import redist

__all__ = [
  'complete_conv_kwargs',
  'complete_deconv_kwargs',
  'get_deconv_kwargs',
  'adjust_channels',
  'get_kernels_by_type',
  'chain', 'seq'
]

def complete_conv_kwargs(conv_kwargs):
  conv_kwargs['filter_size'] = conv_kwargs.get('filter_size', (3, 3))
  conv_kwargs['nonlinearity'] = conv_kwargs.get('nonlinearity', nonlinearities.elu)
  conv_kwargs['pad'] = conv_kwargs.get('pad', 'same')

  return conv_kwargs

def complete_deconv_kwargs(deconv_kwargs):
  deconv_kwargs['filter_size'] = deconv_kwargs.get('filter_size', (3, 3))
  deconv_kwargs['nonlinearity'] = deconv_kwargs.get('nonlinearity', nonlinearities.elu)
  deconv_kwargs['crop'] = deconv_kwargs.get('crop', 'same')

  return deconv_kwargs

def get_deconv_kwargs(conv_kwargs):
  deconv_kwargs = conv_kwargs.copy()
  pad = conv_kwargs.get('pad', 'same')

  if 'pad' in deconv_kwargs:
    del deconv_kwargs['pad']

  if pad == 'same':
    deconv_kwargs['crop'] = 'same'
  elif pad == 'valid':
    deconv_kwargs['crop'] = 'full'
  elif pad == 'full':
    deconv_kwargs['crop'] = 'valid'

  return deconv_kwargs

def get_kernels_by_type(net, kernel_type):
  kernels = []

  for l in layers.get_all_layers(net):
    try:
      W = getattr(l, kernel_type)()
      kernels.append(W)
    except:
      pass

  return kernels

def adjust_channels(incoming, target_channels, redist=redist):
  input_channels = layers.get_output_shape(incoming)[1]

  if input_channels != target_channels:
    return redist(
      incoming=incoming,
      num_filters=target_channels,
      name='channel redistribution'
    )
  else:
    return incoming

def seq(incoming, layer_ops):
  net = incoming
  layers = []

  for layer in layer_ops:
    if hasattr(layer, '__iter__'):
      ls = seq(net, layer)
      layers.extend(ls)
      net = ls[-1]
    elif layer is None:
      pass
    else:
      layers.append(net)

  return layers

def chain(incoming, layer_ops):
  net = incoming

  for layer in layer_ops:
    if hasattr(layer, '__iter__'):
      net = chain(net, layer)
    elif layer is None:
      pass
    else:
      net = layer(net)

  return net