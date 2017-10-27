from lasagne import *

__all__ = [
  'conv', 'conv1x1',
  'deconv', 'deconv1x1',
  'upscale',
  'max_pool', 'mean_pool', 'floating_maxpool', 'floating_meanpool',
  'upconv', 'downconv'
]

get_conv_nonlinearity = lambda f=None: nonlinearities.LeakyRectify(0.05) if f is None else f

get_companion_nonlinearity = lambda n_units=None, f=None: \
  nonlinearities.sigmoid if n_units is None or n_units == 1 else nonlinearities.softmax

conv = lambda num_filters, f=None: lambda incoming: layers.Conv2DLayer(
  incoming,
  num_filters=num_filters, filter_size=(3, 3),
  nonlinearity=get_conv_nonlinearity(f),
  pad='valid'
)

conv1x1 = lambda num_filters, f=None: lambda incoming: layers.Conv2DLayer(
  incoming,
  num_filters=num_filters, filter_size=(1, 1),
  nonlinearity=get_conv_nonlinearity(f),
  pad='valid'
)

deconv = lambda num_filters, f=None: lambda incoming: layers.TransposedConv2DLayer(
  incoming,
  num_filters=num_filters, filter_size=(3, 3),
  nonlinearity=get_conv_nonlinearity(f),
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
  pool_size=(pool_size[0] // 2 * 3, pool_size[0] // 2 * 3),
  stride=pool_size,
  pad=(pool_size[0] // 2, pool_size[1] // 2),
  mode='average_inc_pad'
)