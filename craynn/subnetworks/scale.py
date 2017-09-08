from ..layers import concat, scale_to

__all__ = [
  'scale_concat',
  'scale_concat_rev'
]

def scale_concat(incomings, scale_to=scale_to):
  net, target = incomings
  return concat([
    scale_to(net, target),
    target
  ])

def scale_concat_rev(incomings, scale_to=scale_to):
  target, net = incomings
  return concat([
    scale_to(net, target),
    target
  ])