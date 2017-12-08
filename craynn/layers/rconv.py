from lasagne import *
import theano.tensor as T

__all__ = [
  'RestrictedConv2DLayer',
  'rconv', 'rconv1x1'
]

from .conv_ops import get_conv_nonlinearity

class RestrictedConv2DLayer(layers.Conv2DLayer):
  """
  Just convolution layer with pad='same'.
  """
  def __init__(self, incoming, num_filters, filter_size,
               untie_biases=False,
               W=init.GlorotUniform(1.0),
               b=init.Constant(0.),
               nonlinearity=nonlinearities.LeakyRectify(0.1), flip_filters=True,
               normalization='global', eps=1.0e-6,
               convolution=T.nnet.conv2d, **kwargs):
    stride = (1, 1)
    pad = 'same'

    self.eps = eps

    self.normalization = normalization

    if normalization not in ['unit', 'global']:
      raise ValueError("normalization must be one of ['unit', 'global']")

    super(RestrictedConv2DLayer, self).__init__(incoming, num_filters, filter_size,
                                           stride, pad, untie_biases, W, b,
                                           nonlinearity, flip_filters, convolution,
                                           **kwargs)

  def restricted_conv_kernel(self):
    return self.W

  def restricted_kernel(self):
    return self.W

  def convolve(self, input, **kwargs):
    border_mode = 'half' if self.pad == 'same' else self.pad

    normalized_W = None

    if self.normalization == 'global':
      W_norm = T.sqrt(T.sum(self.W ** 2) + self.eps)
      normalized_W = self.W / W_norm
    elif self.normalization == 'unit':
      W_norm = T.sqrt(T.sum(self.W ** 2, axis=(1, 2, 3)) + self.eps)
      normalized_W = self.W / W_norm[:, None, None, None]

    conved = self.convolution(input, normalized_W,
                              self.input_shape, self.get_W_shape(),
                              subsample=self.stride,
                              border_mode=border_mode,
                              filter_flip=self.flip_filters)
    return conved


rconv = lambda num_filters, f=None, filter_size=(3, 3), normalization='global': lambda incoming: RestrictedConv2DLayer(
  incoming=incoming,
  num_filters=num_filters,
  filter_size=filter_size,
  nonlinearity=get_conv_nonlinearity(f),
  normalization='global'
)

rconv1x1 = lambda num_filters, f=None, normalization='global': lambda incoming: RestrictedConv2DLayer(
  incoming=incoming,
  num_filters=num_filters,
  filter_size=(1, 1),
  nonlinearity=get_conv_nonlinearity(f),
  normalization='global'
)