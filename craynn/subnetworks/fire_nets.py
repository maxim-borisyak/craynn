from .. import layers as clayers

__all__ = [
  'fire_module', 'fire',
  'freeze_module', 'freeze',
  'make_fire_module', 'make_freeze_module'
]

def make_fire_module(
  incoming,
  num_filters = 64,
  squeeze=clayers.squeeze,
  expand=(clayers.diff, clayers.diff1x1),
  merge=clayers.concat()
):

  net = squeeze(num_filters // 4)(incoming)
  expanded = [ exp(num_filters)(net) for exp in expand ]
  return merge(expanded)

fire_module = lambda num_filters, squeeze=clayers.squeeze, expand=(clayers.diff, clayers.diff1x1), merge=clayers.concat(): lambda incoming: \
  make_fire_module(incoming, num_filters, squeeze, expand, merge)

fire = lambda num_filters: lambda incoming: \
  make_fire_module(incoming, num_filters)

def make_freeze_module(
  incoming,
  num_filters = 64,
  squeeze_op=clayers.squeeze,
  expand_ops=(clayers.diff, clayers.diff1x1),
  merge=clayers.concat()
):
  expanded = [ exp(num_filters)(incoming) for exp in expand_ops]
  net = merge(expanded)
  return squeeze_op(num_filters // 4)(net)

freeze_module = lambda num_filters, squeeze_op=clayers.squeeze, expand_ops=(clayers.diff, clayers.diff1x1), merge=clayers.concat(): lambda incoming: \
  make_freeze_module(incoming, num_filters, squeeze_op, expand_ops, merge)

freeze = lambda num_filters: lambda incoming: \
  make_freeze_module(incoming, num_filters)