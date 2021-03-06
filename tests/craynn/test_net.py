import unittest

from lasagne import *
from craynn.layers import *
from craynn.subnetworks import *
from craynn.networks import *

from craynn.viz import draw_to_file

class TestChains(unittest.TestCase):
  def test_import(self):

    downblock = lambda num_filters: achain(diff(num_filters), max_pool())
    upblock = lambda num_filters: achain(upconv(), diff(num_filters))


    nn = UNet((None, 1, 256, 256))(
      [downblock(n) for n in (32, 64, 128)],
      [upblock(n) for n in (32, 64, 128)[::-1]]
    )

    print(nn.inputs)
    print(nn.outputs)
    print(nn.description())

    draw_to_file(layers.get_all_layers(nn.outputs), 'test_net.png')


if __name__ == '__main__':
  unittest.main()
