from lasagne import *
import theano.tensor as T

import numpy as np

__all__ = [
  'BlockConv2D',
  'blockconv', 'blockconv1x1'
]

from .conv_ops import get_conv_nonlinearity

class BlockConv2D(layers.Conv2DLayer):
  """
  Just convolution layer with pad='same'.
  """
  def __init__(self, incoming, num_filters, filter_size,
               connections_per_filter=0.25,
               connection_type='random',
               stride=(1, 1),
               pad='valid',
               untie_biases=False,
               W=init.GlorotUniform(1.0),
               b=init.Constant(0.),
               nonlinearity=nonlinearities.LeakyRectify(0.1),
               convolution=T.nnet.conv2d, **kwargs):

    flip_filters = True
    super(BlockConv2D, self).__init__(
      incoming, num_filters, filter_size,
      stride, pad, untie_biases, W, b,
      nonlinearity, flip_filters, convolution,
      **kwargs
    )

    if connections_per_filter < 1 and connections_per_filter > 0:
      self.connectivity = int(connections_per_filter * self.input_shape[1])
    else:
      self.connectivity = connections_per_filter


    M = np.zeros((num_filters, self.input_shape[1]), dtype='float32')

    if connection_type == 'random':
      for i in range(num_filters):
        indx = np.random.permutation(self.input_shape[1])[:self.connectivity]
        M[i, indx] = 1
    elif connection_type == 'seq' or connection_type == 'sequential':
      for i in range(num_filters):
        indx = np.arange(i, i + connections_per_filter) % self.input_shape[1]
        M[i, indx] = 1

    self.M = self.add_param(M, shape=M.shape, name='mask', trainable=False, regularizable=False)

  def convolve(self, input, **kwargs):
    border_mode = 'half' if self.pad == 'same' else self.pad

    W = self.W * self.M[:, :, None, None]

    conved = self.convolution(input, W,
                              self.input_shape, self.get_W_shape(),
                              subsample=self.stride,
                              border_mode=border_mode,
                              filter_flip=self.flip_filters)
    return conved

  def num_params(self):
    return np.sum(self.M.get_value()) * np.prod(self.filter_size) + (1 if self.untie_biases else self.num_filters)

blockconv = lambda num_filters, f=None, filter_size=(3, 3), connectivity=0.1, connection_type='random': lambda incoming: BlockConv2D(
  incoming=incoming,
  num_filters=num_filters,
  filter_size=filter_size,
  connections_per_filter=connectivity,
  connection_type=connection_type,
  nonlinearity=get_conv_nonlinearity(f),
)

blockconv1x1 = lambda num_filters, f=None, connectivity=0.1, connection_type='random': lambda incoming: BlockConv2D(
  incoming=incoming,
  num_filters=num_filters,
  filter_size=(1, 1),
  connections_per_filter=connectivity,
  connection_type=connection_type,
  nonlinearity=get_conv_nonlinearity(f),
)