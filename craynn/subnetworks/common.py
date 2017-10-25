from lasagne import *
from ..layers import redist

__all__ = [
  'adjust_channels',
  'get_kernels_by_type',
  'chain', 'achain'
]

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

def _chain(incoming, definition):
  net = incoming

  for layer in definition:
    if hasattr(layer, '__iter__'):
      net = _chain(net, layer)
    elif layer is None:
      pass
    else:
      net = layer(net)

  return net

chain = lambda *definition: lambda incoming: _chain(incoming, definition)

def _achain(incoming, definition):
  net = incoming

  if not hasattr(definition, '__iter__'):
    return definition(net)

  for op in definition:
    if hasattr(op, '__iter__'):
      if type(op) is tuple:
        net = _achain(net, op)
      elif type(op) is list:
        net = [
          _achain(net, o)
          for o in op
        ]
    else:
      net = op(net)

  return net

achain = lambda *definition: lambda incoming: _achain(incoming, definition)
