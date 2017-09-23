import theano.tensor as T
from lasagne import *

__all__ = [
  'conv', 'conv1x1',
  'deconv', 'deconv1x1',
  'upscale',
  'max_pool', 'mean_pool', 'floating_maxpool', 'floating_meanpool'
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
  pool_size=(pool_size[0] / 2 * 3, pool_size[0] / 2 * 3),
  stride=pool_size,
  pad=(pool_size[0] / 2, pool_size[1] / 2)
)

upscale = lambda scale_factor=(2, 2): lambda incoming: layers.Upscale2DLayer(incoming, scale_factor=scale_factor)

mean_pool = lambda pool_size=(2, 2): lambda incoming: layers.Pool2DLayer(incoming, pool_size=pool_size, mode='average_inc_pad')

floating_meanpool = lambda pool_size=(2, 2): lambda incoming: layers.Pool2DLayer(
  incoming,
  pool_size=(pool_size[0] / 2 * 3, pool_size[0] / 2 * 3),
  stride=pool_size,
  pad=(pool_size[0] / 2, pool_size[1] / 2),
  mode='average_inc_pad'
)


def concat_conv(incoming1, incoming2, nonlinearity=nonlinearities.elu, name=None,
                W=init.GlorotUniform(0.5),
                avoid_concat=False, *args, **kwargs):
  if avoid_concat:
    conv1 = layers.Conv2DLayer(
      incoming1, nonlinearity=nonlinearities.identity,
      name='%s [part 1]' % (name or ''),
      W = W,
      *args, **kwargs
    )

    conv2 = layers.Conv2DLayer(
      incoming2, nonlinearity=nonlinearities.identity,
      name='%s [part 2]' % (name or ''),
      W=W,
      *args, **kwargs
    )

    u = layers.NonlinearityLayer(
      layers.ElemwiseSumLayer([conv1, conv2], name='%s [sum]' % (name or '')),
      nonlinearity=nonlinearity,
      name='%s [nonlinearity]' % (name or '')
    )

    return u
  else:
    concat = layers.ConcatLayer(
      [incoming1, incoming2], name='%s [concat]' % (name or ''),
      cropping=[None, None, 'center', 'center']
    )

    return layers.Conv2DLayer(
      concat,
      nonlinearity=nonlinearity,
      name='%s [conv]' % (name or ''),
      *args, **kwargs
    )