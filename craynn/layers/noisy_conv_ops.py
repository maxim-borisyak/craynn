import theano.tensor as T

from lasagne.layers import Conv2DLayer
from lasagne import init
from lasagne import nonlinearities

from theano.sandbox.rng_mrg import MRG_RandomStreams

from .conv_ops import get_conv_nonlinearity

__all__ = [
  'NoisyConv2DLayer',
  'NoisyDiff2DLayer',
  'noisy_conv', 'noisy_conv1x1',
  'noisy_diff', 'noisy_diff1x1'
]

class NoisyConv2DLayer(Conv2DLayer):
  """
  Performs Convolution with noisy kernel `conv(W + eps, X)`
  where `eps` is normally distributed with std=`noise_std`.
  """
  def __init__(self, incoming, num_filters, filter_size, noise_std=0.01, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.LeakyRectify(0.05), flip_filters=True,
                 convolution=T.nnet.conv2d, seed=12347, **kwargs):
    super(NoisyConv2DLayer, self).__init__(incoming, num_filters, filter_size,
                                           stride, pad, untie_biases, W, b,
                                           nonlinearity, flip_filters,
                                           convolution,
                                           **kwargs)

    self.noise_std = noise_std
    self.srng = MRG_RandomStreams(seed=seed)

  def get_output_for(self, input, deterministic=False, **kwargs):
    conved = self.convolve(input, deterministic=deterministic, **kwargs)

    if self.b is None:
      activation = conved
    elif self.untie_biases:
      activation = conved + T.shape_padleft(self.b, 1)
    else:
      activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)

    return self.nonlinearity(activation)

  def convolve(self, input, deterministic=False, **kwargs):
    border_mode = 'half' if self.pad == 'same' else self.pad

    if deterministic:
      W = self.W
    else:
      W = self.W + self.srng.normal(size=self.W.get_value(borrow=True).shape, std=self.noise_std)

    conved = self.convolution(input, W,
                              self.input_shape, self.get_W_shape(),
                              subsample=self.stride,
                              border_mode=border_mode,
                              filter_flip=self.flip_filters)
    return conved

noisy_conv = lambda num_filters, f=None, noise_std=1.0e-2: lambda incoming: NoisyConv2DLayer(
  incoming,
  num_filters=num_filters, filter_size=(3, 3),
  noise_std=noise_std,
  nonlinearity=get_conv_nonlinearity(f),
  pad='valid'
)

noisy_conv1x1 = lambda num_filters, f=None, noise_std=1.0e-2: lambda incoming: NoisyConv2DLayer(
  incoming,
  num_filters=num_filters, filter_size=(1, 1),
  noise_std=noise_std,
  nonlinearity=get_conv_nonlinearity(f),
  pad='valid'
)

class NoisyDiff2DLayer(NoisyConv2DLayer):
  def __init__(self, incoming, num_filters, filter_size, noise_std=0.01, stride=(1, 1),
                 untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.LeakyRectify(0.05), flip_filters=True,
                 convolution=T.nnet.conv2d, seed=12347, **kwargs):
    pad = 'same'
    super(NoisyDiff2DLayer, self).__init__(incoming, num_filters, filter_size,
                                           noise_std,
                                           stride, pad, untie_biases, W, b,
                                           nonlinearity, flip_filters,
                                           convolution, seed,
                                           **kwargs)

noisy_diff = lambda num_filters, f=None, noise_std=1.0e-2: lambda incoming: NoisyDiff2DLayer(
  incoming,
  num_filters=num_filters, filter_size=(3, 3),
  noise_std=noise_std,
  nonlinearity=get_conv_nonlinearity(f),
)

noisy_diff1x1 = lambda num_filters, f=None, noise_std=1.0e-2: lambda incoming: NoisyDiff2DLayer(
  incoming,
  num_filters=num_filters, filter_size=(1, 1),
  noise_std=noise_std,
  nonlinearity=get_conv_nonlinearity(f),
)