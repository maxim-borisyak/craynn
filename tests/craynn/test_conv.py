import unittest

import craynn
from craynn.viz import draw_to_file
from craynn.layers import *
from craynn.subnetworks import *
from craynn.networks import *

from lasagne import *

class TestConvNet(unittest.TestCase):
  def test_net(self):

    nn = net((None, 1, 128, 128))(
      conv(12),
      conv1x1(12),
      rconv(12),
      rconv1x1(12),
      blockconv(12),
      blockconv1x1(12),
      diff(12),
      diff1x1(12),
      blockdiff(12),
      blockdiff1x1(12),
    )

    print(
      layers.get_output_shape(nn.outputs)
    )

    print(nn.description())

    print(nn.total_number_of_parameters())

    draw_to_file(layers.get_all_layers(nn.outputs), 'test.png')



if __name__ == '__main__':
  unittest.main()
