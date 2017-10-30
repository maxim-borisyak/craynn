from ..layers import *
from ..subnetworks import unet as unet_factory

from . import *

### To avoid clash of names, save UNet (with upper case letters) as symbol for Network,
### and keep nice syntax, this function is imported as UNet by the networks package.
def unet_(incoming):
  return lambda down_ops, up_ops, concat=concat(): \
    UNet(incoming, down_ops, up_ops, concat)

class UNet(Net):
  def __init__(self, inputs, down_ops, up_ops, concat):
    super(UNet, self).__init__(
      unet_factory(down_ops, up_ops, concat),
      inputs
    )