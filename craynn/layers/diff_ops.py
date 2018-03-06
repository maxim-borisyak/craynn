from lasagne import *
import theano.tensor as T

__all__ = [
  'Diffusion2DLayer',
  'ChannelPooling2DLayer',
  'diff', 'diff1x1',
  'channel_pool', 'channel_factor_pool',
  'diff1D', 'diff1D_1x1'
]

from .conv_ops import get_conv_nonlinearity

class Diffusion2DLayer(layers.Conv2DLayer):
  """
  Just convolution layer with pad='same'.
  """
  def __init__(self, incoming, num_filters, filter_size,
               untie_biases=False,
               W=init.GlorotUniform(1.0),
               b=init.Constant(0.),
               stride=(1, 1),
               nonlinearity=nonlinearities.LeakyRectify(0.05), flip_filters=True,
               convolution=T.nnet.conv2d, **kwargs):
    pad = 'same'
    super(Diffusion2DLayer, self).__init__(incoming, num_filters, filter_size,
                                           stride, pad, untie_biases, W, b,
                                           nonlinearity, flip_filters, convolution,
                                           **kwargs)

  def diffusion_kernel(self):
    return self.W

diff = lambda num_filters, f=None, filter_size=(3, 3), stride=(1, 1): lambda incoming: Diffusion2DLayer(
  incoming=incoming,
  num_filters=num_filters,
  filter_size=filter_size,
  stride=stride,
  nonlinearity=get_conv_nonlinearity(f),
)

double_diff = lambda num_filters, f=None, filter_size=(3, 3): lambda incoming: \
  diff(num_filters, f, filter_size=filter_size)(diff(num_filters, f, filter_size=filter_size)(incoming))

diff1x1 = lambda num_filters, f=None: lambda incoming: Diffusion2DLayer(
  incoming=incoming,
  num_filters=num_filters,
  filter_size=(1, 1),
  nonlinearity=get_conv_nonlinearity(f),
)

class TransposedDiff2DLayer(layers.TransposedConv2DLayer):
  """
  Just convolution layer with pad='same'.
  """
  def __init__(self, incoming, num_filters, filter_size,
               untie_biases=False,
               W=init.GlorotUniform(1.0),
               b=init.Constant(0.),
               stride=(1, 1),
               nonlinearity=nonlinearities.LeakyRectify(0.05), flip_filters=True,
               convolution=T.nnet.conv2d, **kwargs):
    crop = 'same'
    super(TransposedDiff2DLayer, self).__init__(incoming, num_filters, filter_size,
                                                stride, crop, untie_biases, W, b,
                                                nonlinearity, flip_filters, convolution,
                                                **kwargs)

  def diffusion_kernel(self):
    return self.W

dediff = lambda num_filters, f=None, filter_size=(3, 3), stride=(1, 1): lambda incoming: TransposedDiff2DLayer(
  incoming=incoming,
  num_filters=num_filters,
  filter_size=filter_size,
  stride=(1, 1),
  nonlinearity=get_conv_nonlinearity(f)
)

class ChannelPooling2DLayer(layers.Conv2DLayer):
  def __init__(self, incoming, num_filters,
               untie_biases=False,
               W=init.GlorotUniform(1.0),
               nonlinearity=nonlinearities.linear,
               convolution=T.nnet.conv2d, **kwargs):
    stride = (1, 1)
    pad = 'valid'
    filter_size = (1, 1)
    flip_filters = True
    b = None
    super(ChannelPooling2DLayer, self).__init__(incoming, num_filters, filter_size,
                                                stride, pad, untie_biases, W, b,
                                                nonlinearity, flip_filters, convolution,
                                                **kwargs)

  def channel_pooling_kernel(self):
    return self.W

channel_pool = lambda num_filters: lambda incoming: ChannelPooling2DLayer(
  incoming=incoming,
  num_filters=num_filters,
  nonlinearity=nonlinearities.linear,
)

channel_factor_pool = lambda channel_factor=2: lambda incoming: ChannelPooling2DLayer(
  incoming=incoming,
  num_filters=layers.get_output_shape(incoming)[1] // channel_factor,
  nonlinearity=nonlinearities.linear,
)

class Diffusion1DLayer(layers.Conv1DLayer):
  """
  Just convolution layer with pad='same'.
  """
  def __init__(self, incoming, num_filters, filter_size,
               untie_biases=False,
               W=init.GlorotUniform(1.0),
               b=init.Constant(0.),
               stride=(1,),
               nonlinearity=nonlinearities.LeakyRectify(0.05), flip_filters=True,
               **kwargs):
    pad = 'same'
    super(Diffusion1DLayer, self).__init__(incoming, num_filters, filter_size,
                                           stride, pad, untie_biases, W, b,
                                           nonlinearity, flip_filters,
                                           **kwargs)

  def diffusion_kernel(self):
    return self.W

diff1D = lambda num_filters, f=None, filter_size=(3, ), stride=(1, ): lambda incoming: Diffusion1DLayer(
  incoming=incoming,
  num_filters=num_filters,
  filter_size=filter_size,
  stride=stride,
  nonlinearity=get_conv_nonlinearity(f),
)

diff1D_1x1 = lambda num_filters, f=None: lambda incoming: Diffusion1DLayer(
  incoming=incoming,
  num_filters=num_filters,
  filter_size=(1, ),
  nonlinearity=get_conv_nonlinearity(f),
)
