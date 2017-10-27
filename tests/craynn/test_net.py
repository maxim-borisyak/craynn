import unittest

from lasagne import *
from craynn.layers import *
from craynn.subnetworks import *

from craynn.viz import draw_to_file

class TestChains(unittest.TestCase):
  def test_import(self):

    test_input = layers.InputLayer(shape=(None, 1, 256, 256))
    downblock = lambda n_filters: achain(diff(n_filters), max_pool())
    upblock = lambda n_filters: achain(upconv(), diff(n_filters))


    net = unet(
      [downblock(n) for n in (32, 64, 128)],
      [upblock(n) for n in (32, 64, 128)[::-1]]
    )(test_input)

    draw_to_file(layers.get_all_layers(net), 'test_net.png')


if __name__ == '__main__':
  unittest.main()
