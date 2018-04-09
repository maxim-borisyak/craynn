import tensorflow as tf

from ..layers_meta import *
from .conv_common import *

from ...meta import based_on, curry_incoming

__all__ = [
  'MaxPool1DLayer',
  'MaxPool2DLayer',
  'MaxPool3DLayer',

  'MeanPool1DLayer',
  'MeanPool2DLayer',
  'MeanPool3DLayer',

  'max_pool_1d',
  'max_pool_2d',
  'max_pool_3d',

  'mean_pool_1d',
  'mean_pool_2d',
  'mean_pool_3d',

  'max_pool',
  'mean_pool'
]

def _wrapped_pool_1D(op):
  def wrapped_op(X, ksize, strides, padding, data_format, name=None):
    return tf.squeeze(
        op(
          tf.expand_dims(X, axis=-1),
          ksize=ksize + (1, ),
          strides=strides + (1, ),
          padding=padding,
          data_format=data_format,
          name=name
      ),
      axis=-1
    )
  return wrapped_op

class PoolLayer(FunctionalLayer):
  def __init__(self, incoming, ndim, pool_op, pool_size, stride=None, pad='same', name=None):
    super(PoolLayer, self).__init__(incoming, name=name)

    self.ndim = ndim

    self.pool_size = get_kernel_size(pool_size, ndim=ndim)
    self.channel_pool = 1
    self.kernel_size = (1, ) + self.pool_size + (self.channel_pool, )

    self.stride = get_stride(self.pool_size if stride is None else stride, ndim=ndim)
    self.pad = get_pad(pad)

    self.pool_op = pool_op

  def get_output_for(self, X):
    return self.pool_op(
      X,
      ndim=self.ndim,
      ksize=self.kernel_size,
      padding=self.pad,
      strides=self.stride,
      data_format=strange_data_format(self.ndim)
    )

  def get_output_shape_for(self, input_shape):
    return (input_shape[0], ) + get_kernel_output_shape(
      input_shape[1:-1],
      spatial_kernel_size=self.kernel_size[1:-1],
      pad=self.pad,
      stride=self.stride,
    ) + (input_shape[-1], )

MeanPool1DLayer = based_on(PoolLayer).derive('MeanPool1DLayer').let(
  ndim=1,
  pool_op=_wrapped_pool_1D(tf.nn.avg_pool)
).with_defaults(
  pool_size=(2, )
)

MaxPool1DLayer = based_on(PoolLayer).derive('MaxPool1DLayer').let(
  ndim=1,
  pool_op=_wrapped_pool_1D(tf.nn.max_pool)
).with_defaults(
  pool_size=(2, )
)

MeanPool2DLayer = based_on(PoolLayer).derive('MeanPool2DLayer').let(
  ndim=2,
  pool_op=tf.nn.avg_pool
).with_defaults(
  pool_size=(2, 2)
)

MaxPool2DLayer = based_on(PoolLayer).derive('MaxPool2DLayer').let(
  ndim=2,
  pool_op=tf.nn.max_pool
).with_defaults(
  pool_size=(2, 2)
)

MeanPool3DLayer = based_on(PoolLayer).derive('MeanPool3DLayer').let(
  ndim=3,
  pool_op=tf.nn.avg_pool3d
).with_defaults(
  pool_size=(2, 2, 2)
)

MaxPool3DLayer = based_on(PoolLayer).derive('MaxPool3DLayer').let(
  ndim=3,
  pool_op=tf.nn.max_pool3d
).with_defaults(
  pool_size=(2, 2, 2)
)

mean_pool_1d = curry_incoming(MeanPool1DLayer, name='mean_pool_1d')
max_pool_1d = curry_incoming(MaxPool1DLayer, name='max_pool_1d')

mean_pool_2d = curry_incoming(MeanPool2DLayer, name='mean_pool_2d')
max_pool_2d = curry_incoming(MaxPool2DLayer, name='max_pool_2d')

mean_pool = mean_pool_2d
max_pool = max_pool_2d

mean_pool_3d = curry_incoming(MeanPool3DLayer, name='mean_pool_3d')
max_pool_3d = curry_incoming(MaxPool3DLayer, name='max_pool_3d')