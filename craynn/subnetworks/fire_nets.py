import numpy as np
import theano
import theano.tensor as T
from lasagne import *

from .. import layers as clayers

__all__ = [
  'fire_module', 'fire',
  'freeze_module', 'freeze',
  'make_fire_module', 'make_freeze_module'
]

def make_fire_module(
  incoming,
  n_filters = 64,
  squeeze=clayers.squeeze,
  expand=(clayers.diff, clayers.diff1x1),
  merge=clayers.concat
):

  net = squeeze(n_filters / 4)(incoming)
  expanded = [ exp(n_filters)(net) for exp in expand ]
  return merge()(expanded)

fire_module = lambda n_filters, squeeze=clayers.squeeze, expand=(clayers.diff, clayers.diff1x1), merge=clayers.concat: lambda incoming: \
  make_fire_module(incoming, n_filters, squeeze, expand, merge)

fire = lambda n_filters: lambda incoming: \
  make_fire_module(incoming, n_filters)

def make_freeze_module(
  incoming,
  n_filters = 64,
  squeeze_op=clayers.squeeze,
  expand_ops=(clayers.diff, clayers.diff1x1),
  merge=clayers.concat()
):
  expanded = [ exp(n_filters)(incoming) for exp in expand_ops]
  net = merge(expanded)
  return squeeze_op(n_filters / 4)(net)

freeze_module = lambda n_filters, squeeze_op=clayers.squeeze, expand_ops=(clayers.diff, clayers.diff1x1), merge=clayers.concat(): lambda incoming: \
  make_freeze_module(incoming, n_filters, squeeze_op, expand_ops, merge)

freeze = lambda n_filters: lambda incoming: \
  make_freeze_module(incoming, n_filters)