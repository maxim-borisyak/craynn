from lasagne import *
import theano.tensor as T

__all__ = [
  'Diffusion2DLayer',
  'Redistribution2DLayer',
  'diff',
  'diff1x1',
  'redist'
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
               nonlinearity=nonlinearities.LeakyRectify(0.05), flip_filters=True,
               convolution=T.nnet.conv2d, **kwargs):
    stride = (1, 1)
    pad = 'same'
    super(Diffusion2DLayer, self).__init__(incoming, num_filters, filter_size,
                                           stride, pad, untie_biases, W, b,
                                           nonlinearity, flip_filters, convolution,
                                           **kwargs)

  def diffusion_kernel(self):
    return self.W

diff = lambda num_filters, f=None: lambda incoming: Diffusion2DLayer(
  incoming=incoming,
  num_filters=num_filters,
  filter_size=(3, 3),
  nonlinearity=get_conv_nonlinearity(f),
)

diff1x1 = lambda num_filters, f=None: lambda incoming: Diffusion2DLayer(
  incoming=incoming,
  num_filters=num_filters,
  filter_size=(1, 1),
  nonlinearity=get_conv_nonlinearity(f),
)

class Redistribution2DLayer(layers.Conv2DLayer):
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
    super(Redistribution2DLayer, self).__init__(incoming, num_filters, filter_size,
                                                stride, pad, untie_biases, W, b,
                                                nonlinearity, flip_filters, convolution,
                                                **kwargs)

  def redistribution_kernel(self):
    return self.W

redist = lambda num_filters: lambda incoming: Redistribution2DLayer(
  incoming=incoming,
  num_filters=num_filters,
  nonlinearity=nonlinearities.linear,
)