import unittest

from lasagne import *
from craynn.layers import *
from craynn.subnetworks import *
from craynn.networks import *

from craynn.viz import draw_to_file

class TestEBNet(unittest.TestCase):
  def test_energy_based(self):

    nn = EnergyBased((None, 1, 38, 38))(mask(9), mask(9)) (
      conv(16), max_pool(),
      conv(32), max_pool(),
      conv(64),
      deconv(32), upscale(),
      deconv(16), upscale(),
      deconv(1),
    )

    print(nn.inputs)
    print(nn.outputs)
    print(nn.description())

    draw_to_file(layers.get_all_layers(nn.outputs), 'test_net.png')


if __name__ == '__main__':
  unittest.main()
