import tensorflow as tf

from .layers_meta import FunctionalLayer, get_output_shape
from .common_ops import get_nonlinearity, default_common_nonlinearity
from .init import normal, zeros

__all__ = [
  'DenseLayer',
  'dense'
]

class DenseLayer(FunctionalLayer):
  def __init__(self, incoming, num_units,
               nonlinearity=default_common_nonlinearity,
               W=normal(0.0, 1.0e-3),
               b=zeros(),
               name=None):
    super(DenseLayer, self).__init__(incoming, name=name)
    input_shape = get_output_shape(incoming)
    self.num_units = num_units

    if len(input_shape) != 2:
      raise ValueError('Dense layer accepts only 2D tensors!')

    self.W = tf.Variable(
      initial_value=W((input_shape[1], num_units)),
      dtype='float32',
      expected_shape=(input_shape[1], num_units),
    )

    self.b = tf.Variable(
      initial_value=b((num_units, )),
      dtype='float32',
      expected_shape=(num_units, )
    )

    self.params['W'] = (self.W, ['weights', 'trainable', 'free'])
    self.params['b'] = (self.b, ['biases', 'trainable', 'free'])

    self.nonlinearity = get_nonlinearity(nonlinearity)

  def get_output_for(self, X):
    return self.nonlinearity(
      tf.matmul(X, self.W) + self.b[None, :]
    )

  def get_output_shape_for(self, input_shape):
    if len(input_shape) != 2:
      raise ValueError('Dense layer accepts only 2D tensors!')

    return (input_shape[0], self.num_units)

dense = lambda num_units, f='default', name=None: lambda incoming: \
  DenseLayer(incoming, num_units, nonlinearity=f, name=name)
