from ..layers import *

from lasagne import layers

__all__ = [
  'unet'
]

def _unet(incoming, down_ops, up_ops, concat=concat()):
  down_chain = []

  net = incoming

  for op in down_ops:
    net = op(net)
    down_chain.append(net)

  op = up_ops[0]
  net = op(net)

  for down_layer, op in zip(down_chain[:-1][::-1], up_ops[1:]):
    net = op(concat([net, down_layer]))

  return net

unet = lambda down_ops, up_ops, concat=concat(): lambda incoming: \
  _unet(incoming, down_ops, up_ops, concat)
