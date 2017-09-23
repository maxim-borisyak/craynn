import unittest

from lasagne import *
from craynn.layers import *
from craynn.subnetworks import *

from craynn.viz import draw_to_file

class TestChains(unittest.TestCase):
  def test_import(self):

    test_input = layers.InputLayer(shape=(None, 1, 128, 128))

    net = chain(
      (conv(4), max_pool()),
      (conv(2), max_pool()),
      (conv(3), max_pool()),
      (conv(6), max_pool()),
      conv(12),
    )(test_input)

    net = cnn(
      num_filters=(1, 2, 3),
    )(test_input)

    draw_to_file(layers.get_all_layers(net), 'test_net.png')


if __name__ == '__main__':
  unittest.main()
