import unittest

from lasagne import *
from craynn.layers import *
from craynn.subnetworks import *
from craynn.networks import *

from craynn.viz import draw_to_file

class TestChains(unittest.TestCase):
  def test_import(self):

    test_input = layers.InputLayer(shape=(None, 1, 256, 256))
    downblock = lambda n_filters: achain(diff(n_filters), max_pool())
    upblock = lambda n_filters: achain(upconv(), diff(n_filters))


    factory = unet(
      [downblock(n) for n in (32, 64, 128)],
      [upblock(n) for n in (32, 64, 128)[::-1]]
    )

    #nn = net(test_input)(factory)
    nn = net((None, 1, 256, 256))(factory)
    print(nn.inputs)
    print(nn.outputs)
    print(nn.description())

    draw_to_file(layers.get_all_layers(nn.outputs), 'test_net.png')


if __name__ == '__main__':
  unittest.main()
