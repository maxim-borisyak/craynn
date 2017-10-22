from ..layers import *

__all__ = [
  'downprop', 'upprop'
]

def get_origins(incomings, depth, down=max_pool()):
  if not hasattr(incomings, '__iter__'):
    origins = [incomings]

    for i in range(depth - 1):
      origins.append(
        down(origins[-1])
      )
    return origins
  else:
    return incomings

def make_downprop_block(incomings, base_blocks, up=upscale(), down=max_pool(), concat=concat()):
  origins = get_origins(incomings, depth=len(base_blocks), down=down)
  outputs = []

  net = None
  for origin, base in zip(origins[::-1], base_blocks):
    if net is None:
      net = base(origin)
    else:
      net = concat(up(net), origin)
      net = base(net)

    outputs.append(net)

  return outputs

downprop = lambda base_blocks, up=upscale(), down=max_pool(), concat=concat(): \
  lambda incomings: make_downprop_block(incomings, base_blocks, up, down, concat)

def make_upprop_block(incomings, base_blocks, down=max_pool(), concat=concat()):
  origins = get_origins(incomings, depth=len(base_blocks), down=down)
  outputs = []

  net = None
  for origin, base in zip(origins, base_blocks[::-1]):
    if net is None:
      net = base(origin)
    else:
      net = concat(down(net), origin)
      net = base(net)

    outputs.append(net)

  return outputs


upprop = lambda base_blocks, down=max_pool(), concat=concat(): \
  lambda incomings: make_upprop_block(incomings, base_blocks, down, concat)







