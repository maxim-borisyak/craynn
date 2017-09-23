from lasagne import layers

__all__ = [
  'min', 'max',
  'concat',
  'noise', 'nothing'
]

min = lambda: lambda incomings: layers.ElemwiseMergeLayer(incomings, merge_function=T.minimum)
max = lambda: lambda incomings: layers.ElemwiseMergeLayer(incomings, merge_function=T.maximum)
concat = lambda: lambda incomings: layers.ConcatLayer(incomings)

noise = lambda sigma=0.1: lambda incoming: layers.GaussianNoiseLayer(incoming, sigma=sigma)
nothing = lambda incoming: incoming