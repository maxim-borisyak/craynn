import unittest

import craynn
from craynn.viz import draw_to_file
from craynn.layers import *
from craynn.subnetworks import *
from craynn.networks import *

from lasagne import *

class TestRepeat(unittest.TestCase):
  def test_net(self):
    nn = net((None, 1, 128, 128))(
      repeat(8)(conv(4))
    )

    print(
      layers.get_output_shape(nn.outputs)
    )

    draw_to_file(layers.get_all_layers(nn.outputs), 'test.png')



if __name__ == '__main__':
  unittest.main()
