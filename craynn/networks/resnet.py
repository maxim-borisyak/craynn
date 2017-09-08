from crayimage.nn import Expression
from lasagne import *

from common import *
from craynn.subnetworks import make_resnet, make_resae

__all__ = [
  'ResNet',
  'ResNetAE'
]

class ResNet(Expression):
  def __init__(self,
               block_size, channels = None,
               noise_sigma = 1.0/1024,
               img_shape = None, input_layer = None, **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net = layers.GaussianNoiseLayer(self.input_layer, sigma = noise_sigma)
    net = make_resnet(net, channels, block_size, **conv_kwargs)
    
    super(ResNet, self).__init__(self.input_layer, [net])

class ResNetAE(Expression):
  def __init__(self,
               block_size, channels = None,
               noise_sigma=1.0 / 1024,
               img_shape=None, input_layer=None,
               **conv_kwargs):
    self.input_layer = get_input_layer(img_shape, input_layer)

    net = layers.GaussianNoiseLayer(self.input_layer, sigma=noise_sigma)
    net = make_resae(net, channels, block_size, **conv_kwargs)

    super(ResNetAE, self).__init__(self.input_layer, [net])
