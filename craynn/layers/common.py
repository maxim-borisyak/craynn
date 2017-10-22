from lasagne import layers

import theano.tensor as T

__all__ = [
  'take', 'min', 'max', 'concat', 'noise', 'nothing', 'batch_norm'
]

min = lambda: lambda incomings: layers.ElemwiseMergeLayer(incomings, merge_function=T.minimum)
max = lambda: lambda incomings: layers.ElemwiseMergeLayer(incomings, merge_function=T.maximum)
concat = lambda axis=1: lambda incomings: layers.ConcatLayer(incomings, axis=axis)

noise = lambda sigma=0.1: lambda incoming: layers.GaussianNoiseLayer(incoming, sigma=sigma)
nothing = lambda incoming: incoming

batch_norm = lambda axes='auto': lambda incoming: layers.BatchNormLayer(incoming, axes=axes)

class Take(object):
  def __getitem__(self, item):
    return lambda incomings: incomings[item]

take = Take()