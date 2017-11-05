from lasagne import layers, nonlinearities

import theano.tensor as T

__all__ = [
  'take', 'minimum', 'maximum', 'concat', 'noise', 'nothing', 'dropout', 'dense',
  'batch_norm', 'elementwise', 'elementwise_sum', 'flatten', 'feature_pool'
]

minimum = lambda: lambda incomings: layers.ElemwiseMergeLayer(incomings, merge_function=T.minimum)
maximum = lambda: lambda incomings: layers.ElemwiseMergeLayer(incomings, merge_function=T.maximum)
concat = lambda axis=1: lambda incomings: layers.ConcatLayer(incomings, axis=axis)

noise = lambda sigma=0.1: lambda incoming: \
  layers.GaussianNoiseLayer(incoming, sigma=sigma) if sigma is not None and sigma > 0 else incoming

nothing = lambda incoming: incoming

dense = lambda num_units, f=None: lambda incoming: \
  layers.DenseLayer(incoming, num_units=num_units, nonlinearity=(nonlinearities.LeakyRectify(0.05) if f is None else f))

dropout = lambda p=0.1, rescale=True: lambda incoming: \
  layers.DropoutLayer(incoming, p=p, rescale=rescale) if p is not None else incoming

batch_norm = lambda axes='auto': lambda incoming: layers.BatchNormLayer(incoming, axes=axes)

class Take(object):
  def __getitem__(self, item):
    return lambda incomings: incomings[item]

take = Take()

elementwise = lambda f=T.add: lambda incomings: layers.ElemwiseMergeLayer(incomings, f)
elementwise_sum = lambda: lambda incomings: layers.ElemwiseMergeLayer(incomings, T.add)

flatten = lambda outdim=2: lambda incoming: layers.FlattenLayer(incoming, outdim=outdim)

feature_pool = lambda pool_size=4, axis=1, f=T.max: lambda incoming: \
  layers.FeaturePoolLayer(incoming, pool_size=pool_size, axis=axis, pool_function=f)
