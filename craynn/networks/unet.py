from .. import Expression
from lasagne import *

from .common import *
from craynn.subnetworks import make_unet

__all__ = [
  'UNet',
]


class UNet(Expression):
  def __init__(self,
               channels=None,
               img_shape=None, input_layer=None, **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net, self.forward, self.backward = make_unet(self.input_layer, channels, return_groups=True, **conv_kwargs)

    super(UNet, self).__init__([self.input_layer], [net])