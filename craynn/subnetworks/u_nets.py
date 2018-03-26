from ..layers import *

from lasagne import layers

__all__ = [
  'unet'
]

def _unet(incoming, down_ops, up_ops, middle_ops=list(), concat=concat()):
  down_chain = []

  net = incoming

  for op in down_ops:
    net = op(net)
    down_chain.append(net)

  for op in middle_ops:
    net = op(net)

  for down_layer, op in zip(down_chain[::-1], up_ops):
    net = op(concat([net, down_layer]))

  return net

unet = lambda down_ops, up_ops, middle_ops=list(), concat=concat(): lambda incoming: \
  _unet(incoming, down_ops, up_ops, middle_ops, concat)
