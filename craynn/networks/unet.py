from ..layers import *
from ..subnetworks import *

from . import *

def unet_(incoming):
  return lambda down_ops, up_ops, concat=concat(): \
    UNet(incoming, down_ops, up_ops, concat)

class UNet(Net):
  def __init__(self, inputs, down_ops, up_ops, concat):
    super(UNet, self).__init__(
      unet(down_ops, up_ops, concat),
      inputs
    )