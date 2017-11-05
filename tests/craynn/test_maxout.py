import unittest

import craynn
from craynn.viz import draw_to_file
from craynn.layers import *
from craynn.subnetworks import *
from craynn.networks import *

from lasagne import *

class TestResnet(unittest.TestCase):
  def test_net(self):
    block = lambda n_filters: maxdropout2d(n_filters)

    nn = net((None, 1, 128, 128))(
      block(32), max_pool(),
      block(64), max_pool(),
      block(128), max_pool(),
    )

    print(
      layers.get_output_shape(nn.outputs)
    )

    draw_to_file(layers.get_all_layers(nn.outputs), 'test.png')



if __name__ == '__main__':
  unittest.main()
