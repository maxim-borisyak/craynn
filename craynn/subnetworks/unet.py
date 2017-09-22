from common import complete_conv_kwargs

from lasagne import *
from ..layers import concat_conv

__all__ = [
  'make_unet',
]

def make_unet(input_layer, convs, deconvs, pool, concat, return_groups = False):
  initial_channels = layers.get_output_shape(input_layer)[1]

  net = input_layer
  forward = [net]
  backward = []

  for i, conv in enumerate(conv_ops[-1]):
    net = conv(net)
    net = pool(net)

    forward.append(net)

  net = conv_ops[-1](net)

  for i, (n_chl, l) in enumerate(zip(n_channels[:-1][::-1], forward[::-1])):
    net = concat_conv(
      net, l,
      num_filters=n_chl,
      **conv_kwargs
    )
    backward.append(net)

    net = layers.Upscale2DLayer(net, scale_factor=(2, 2))

  net = concat_conv(
    net, input_layer,
    num_filters=initial_channels,
    **conv_kwargs
  )
  backward.append(net)

  if return_groups:
    return net, forward, backward
  else:
    return net

