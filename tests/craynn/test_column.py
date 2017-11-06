import unittest

import craynn
from craynn.viz import draw_to_file
from craynn.layers import *
from craynn.subnetworks import *
from craynn.networks import *

from lasagne import *

class TestColumnNet(unittest.TestCase):
  def test_net(self):
    block = lambda num_filters: column(
      num_filters_per_column=num_filters // 4,
      number_of_columns=4,
      column_channels=num_filters // 4
    )

    nn = net((None, 1, 128, 128))(
      block(32), max_pool(),
      block(64), max_pool(),
      block(128), max_pool(),
      companion(num_units=10)
    )

    print(
      layers.get_output_shape(nn.outputs)
    )

    print(nn.total_number_of_parameters())

    draw_to_file(layers.get_all_layers(nn.outputs), 'test.png')



if __name__ == '__main__':
  unittest.main()
