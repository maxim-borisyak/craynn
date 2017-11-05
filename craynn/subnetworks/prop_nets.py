from ..layers import *

__all__ = [
  'downprop', 'upprop'
]

def get_origins(incomings, depth, down_op=max_pool()):
  if not hasattr(incomings, '__iter__'):
    origins = [incomings]

    for i in range(depth - 1):
      origins.append(
        down_op(origins[-1])
      )
    return origins
  else:
    return incomings

def make_downprop_block(incomings, ops, up_op=upscale(), down_op=max_pool(), concat_op=concat()):
  origins = get_origins(incomings, depth=len(ops), down_op=down_op)
  outputs = []

  net = None
  for origin, op in zip(origins[::-1], ops):
    if net is None:
      net = op(origin)
    else:
      net = concat_op([up_op(net), origin])
      net = op(net)

    outputs.append(net)

  return outputs[::-1]

downprop = lambda ops, up_op=upscale(), down_op=max_pool(), concat_op=concat(): \
  lambda incomings: make_downprop_block(incomings, ops, up_op, down_op, concat_op)

def make_upprop_block(incomings, base_blocks, down=max_pool(), concat=concat()):
  origins = get_origins(incomings, depth=len(base_blocks), down_op=down)
  outputs = []

  net = None
  for origin, base in zip(origins, base_blocks[::-1]):
    if net is None:
      net = base(origin)
    else:
      net = concat([down(net), origin])
      net = base(net)

    outputs.append(net)

  return outputs


upprop = lambda base_blocks, down=max_pool(), concat=concat(): \
  lambda incomings: make_upprop_block(incomings, base_blocks, down, concat)







