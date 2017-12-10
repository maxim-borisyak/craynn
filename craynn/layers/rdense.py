import theano
import theano.tensor as T

import lasagne

from lasagne import init
from lasagne import nonlinearities

from .common import get_common_nonlinearity

__all__ = [
  'RestrictedDenseLayer',
  'rdense'
]

class RestrictedDenseLayer(lasagne.layers.DenseLayer):
  def __init__(self, incoming, num_units, W=init.GlorotUniform(),
               b=init.Constant(0.), nonlinearity=nonlinearities.LeakyRectify(0.1),
               num_leading_axes=1,
               normalization='global', eps=1.0e-6,
               **kwargs):
    super(RestrictedDenseLayer, self).__init__(
      incoming, num_units, W, b, nonlinearity, num_leading_axes, **kwargs
    )

    self.normalization = normalization
    self.eps = eps

    if normalization not in ['unit', 'global']:
      raise ValueError("normalization must be one of ['unit', 'global']")

  def restricted_dense_kernel(self):
    return self.W

  def restricted_kernel(self):
    return self.W

  def get_output_for(self, input, **kwargs):
    num_leading_axes = self.num_leading_axes
    if num_leading_axes < 0:
      num_leading_axes += input.ndim
    if input.ndim > num_leading_axes + 1:
      # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
      input = input.flatten(num_leading_axes + 1)

    normalized_W = None

    if self.normalization == 'global':
      W_norm = T.sqrt(T.sum(self.W ** 2) + self.eps)
      normalized_W = self.W / W_norm
    elif self.normalization == 'unit':
      W_norm = T.sqrt(T.sum(self.W ** 2, axis=(0, )) + self.eps)
      normalized_W = self.W / W_norm[None, :]

    activation = T.dot(input, normalized_W)
    if self.b is not None:
      activation = activation + self.b
    return self.nonlinearity(activation)

rdense = lambda num_units, f=None, normalization='global': lambda incoming: RestrictedDenseLayer(
  incoming=incoming,
  num_units=num_units,
  f=get_common_nonlinearity(f),
  normalization=normalization
)