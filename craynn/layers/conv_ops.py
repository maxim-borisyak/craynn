from lasagne import *

import theano.tensor as T

__all__ = [
  'conv', 'conv1x1',
  'deconv', 'deconv1x1',
  'upscale',
  'max_pool', 'mean_pool', 'floating_maxpool', 'floating_meanpool',
  'global_pool',
  'upconv', 'downconv',
  'conv_1D', 'conv1x1_1D', 'max_pool_1D', 'upscale_1D'
]

### common ops

get_conv_nonlinearity = lambda f=None: nonlinearities.LeakyRectify(0.05) if f is None else f

get_companion_nonlinearity = lambda num_units=None, f=None: \
  nonlinearities.sigmoid if num_units is None or num_units == 1 else nonlinearities.softmax

get_pool_function = lambda f=None: T.max if f is None else f

global_pool = lambda f: lambda incoming: layers.GlobalPoolLayer(incoming, pool_function=f)

### 2D ops

conv = lambda num_filters, f=None, filter_size=(3, 3), stride=(1, 1): lambda incoming: layers.Conv2DLayer(
  incoming,
  num_filters=num_filters, filter_size=filter_size,
  nonlinearity=get_conv_nonlinearity(f),
  stride=stride,
  pad='valid'
)

conv1x1 = lambda num_filters, f=None: lambda incoming: layers.Conv2DLayer(
  incoming,
  num_filters=num_filters, filter_size=(1, 1),
  nonlinearity=get_conv_nonlinearity(f),
  pad='valid'
)

deconv = lambda num_filters, f=None, filter_size=(3, 3), stride=(1, 1): lambda incoming: layers.TransposedConv2DLayer(
  incoming,
  num_filters=num_filters, filter_size=filter_size,
  nonlinearity=get_conv_nonlinearity(f),
  stride=stride,
  crop='valid'
)

deconv1x1 = lambda num_filters, f=None: lambda incoming: layers.TransposedConv2DLayer(
  incoming,
  num_filters=num_filters, filter_size=(1, 1),
  nonlinearity=get_conv_nonlinearity(f),
  crop='valid'
)

max_pool = lambda pool_size=(2, 2): lambda incoming: layers.MaxPool2DLayer(incoming, pool_size=pool_size)

floating_maxpool = lambda pool_size=(2, 2): lambda incoming: layers.MaxPool2DLayer(
  incoming,
  pool_size=(pool_size[0] // 2 * 3, pool_size[0] // 2 * 3),
  stride=pool_size,
  pad=(pool_size[0] // 2, pool_size[1] // 2)
)

upscale = lambda scale_factor=(2, 2): lambda incoming: layers.Upscale2DLayer(incoming, scale_factor=scale_factor)
downconv = lambda scale_factor=(2, 2), channel_factor=1: lambda incoming: \
  layers.Conv2DLayer(
    incoming,
    num_filters=layers.get_output_shape(incoming)[1] // channel_factor,
    filter_size=scale_factor,
    stride=scale_factor,
    nonlinearity=nonlinearities.linear
  )

upconv = lambda scale_factor=(2, 2), channel_factor=1: lambda incoming: \
  layers.TransposedConv2DLayer(
    incoming,
    num_filters=layers.get_output_shape(incoming)[1] // channel_factor,
    filter_size=scale_factor,
    stride=scale_factor,
    nonlinearity=nonlinearities.linear
  )

mean_pool = lambda pool_size=(2, 2): lambda incoming: layers.Pool2DLayer(incoming, pool_size=pool_size, mode='average_inc_pad')

floating_meanpool = lambda pool_size=(2, 2): lambda incoming: layers.Pool2DLayer(
  incoming,
  pool_size=(pool_size[0] // 2 * 3, pool_size[1] // 2 * 3),
  stride=pool_size,
  pad=(pool_size[0] // 2, pool_size[1] // 2),
  mode='average_inc_pad'
)

### 1D ops

conv_1D = lambda num_filters, f=None, filter_size=(3,), stride=(1, ): lambda incoming: layers.Conv1DLayer(
  incoming,
  num_filters=num_filters, filter_size=filter_size,
  nonlinearity=get_conv_nonlinearity(f),
  stride=stride,
  pad='valid'
)

conv1x1_1D = lambda num_filters, f=None: lambda incoming: layers.Conv1DLayer(
  incoming,
  num_filters=num_filters, filter_size=(1, ),
  nonlinearity=get_conv_nonlinearity(f),
  pad='valid'
)

max_pool_1D = lambda pool_size=(2,): lambda incoming: layers.MaxPool1DLayer(incoming, pool_size=pool_size)

floating_maxpool_1D = lambda pool_size=(2,): lambda incoming: layers.MaxPool1DLayer(
  incoming,
  pool_size=(pool_size[0] // 2 * 3, ),
  stride=pool_size,
  pad=(pool_size[0] // 2,)
)

upscale_1D = lambda scale_factor=(2, ): lambda incoming: layers.Upscale1DLayer(incoming, scale_factor=scale_factor)
