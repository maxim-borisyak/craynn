import unittest

import craynn
from craynn.viz import draw_to_file
from craynn.layers import *
from craynn.subnetworks import *
from craynn.networks import *

from lasagne import *

class TestPropNet(unittest.TestCase):
  def test_import(self):
    test_input = layers.InputLayer(shape=(None, 1, 28, 28))

    block = lambda n_filters, depth: \
      downprop([ diff(n_filters) for _ in range(depth) ])

    outputs = achain(
      block(4, 3), take[0], max_pool(),
      block(4, 2), take[0], max_pool(),
      block(4, 1), take[0]
    )(test_input)

    print(
      layers.get_output_shape(outputs)
    )

    draw_to_file(layers.get_all_layers(outputs), 'test.png')
    raise Exception()



if __name__ == '__main__':
  unittest.main()
