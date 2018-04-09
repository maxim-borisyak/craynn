import numpy as np

import tensorflow as tf

__all__ = [
  'normal',
  'zeros',

  'default_weight_init',
  'default_bias_init'
]

normal = lambda mean=0.0, std=1.0e-3, dtype='float32': lambda shape: \
  np.random.normal(loc=mean, scale=std, size=shape).astype(dtype)

zeros = lambda dtype='float32': lambda shape: \
  np.zeros(shape=shape, dtype=dtype)

default_weight_init = normal(0.0, 1.0e-3)
default_bias_init = zeros()