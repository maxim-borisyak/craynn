from lasagne import layers

import theano.tensor as T

__all__ = [
  'min', 'max', 'concat', 'noise', 'nothing'
]

min = lambda: lambda incomings: layers.ElemwiseMergeLayer(incomings, merge_function=T.minimum)
max = lambda: lambda incomings: layers.ElemwiseMergeLayer(incomings, merge_function=T.maximum)
concat = lambda axis=1: lambda incomings: layers.ConcatLayer(incomings, axis=axis)

noise = lambda sigma=0.1: lambda incoming: layers.GaussianNoiseLayer(incoming, sigma=sigma)
nothing = lambda incoming: incoming