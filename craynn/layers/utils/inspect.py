from craynn.layers.layers_meta import *

import numpy as np
from collections import OrderedDict


__all__ = [
  'get_number_of_params',
  'get_total_number_of_params'
]

def get_number_of_params(layer, **properties):
  params = get_params(layer, **properties)
  shapes = [ tuple(param.shape.as_list()) for param in params ]
  nums_params = [ np.prod(shape, dtype='int64') for shape in shapes ]

  return OrderedDict(zip(params, nums_params))

def get_total_number_of_params(layer, **properties):
  params = get_params(layer,  **properties)
  if len(params) == 0:
    return None

  shapes = [tuple(param.shape.as_list()) for param in params]
  return np.sum(
    [np.prod(shape, dtype='int64') for shape in shapes],
    dtype='int64'
  )
