import unittest

from lasagne import *
from craynn.layers import *
from craynn.subnetworks import *

from craynn.viz import draw_to_file

class TestChains(unittest.TestCase):
  def test_import(self):

    test_input = layers.InputLayer(shape=(None, 1, 256, 256))

    net = chain(
      (freeze(4), max_pool()),
      (freeze(16), max_pool()),
      (freeze(16), max_pool()),
      (freeze(6), max_pool()),
      freeze(12),
    )(test_input)

    draw_to_file(layers.get_all_layers(net), 'test_net.png')


if __name__ == '__main__':
  unittest.main()
