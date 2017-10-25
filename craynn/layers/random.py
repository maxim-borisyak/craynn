import numpy as np
import lasagne

# from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .meta import AutoInputLayer

__all__ = [
  'GaussianInput'
]

class RandomInput(AutoInputLayer):
  def __init__(self, shape_or_layer, name=None, **kwargs):
    self.srng = RandomStreams(seed=np.random.randint(2147483647))

    if isinstance(shape_or_layer, lasagne.layers.Layer):
      shape=lasagne.layers.get_output_shape(shape_or_layer)
      self.linked_layer = shape_or_layer
    else:
      shape = shape_or_layer
      self.linked_layer = None

    name = 'Random Input' if name is None else name
    super(RandomInput, self).__init__(shape=shape, name=name, **kwargs)

  def generate(self, shape, ndim):
    raise NotImplementedError

  def get_autoinput(self, linked_input=None):
    if linked_input is not None:
      return self.generate(linked_input.shape, ndim=linked_input.ndim)
    else:
      return self.generate(self.shape, ndim=len(self.shape))

class GaussianInput(RandomInput):
  def __init__(self, shape_or_layer, mean=0.0, std=1.0, name=None, **kwargs):
    self.mean = mean
    self.std = std
    
    super(GaussianInput, self).__init__(shape_or_layer, name, **kwargs)

  def generate(self, shape, ndim):
    return self.srng.normal(size=shape, avg=self.mean, std=self.std, ndim=ndim)