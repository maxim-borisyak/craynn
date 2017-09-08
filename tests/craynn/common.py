import unittest

import theano

from craynn import subnetworks

theano.config.floatX = "float32"

from lasagne import *

from craynn import conv

class CommonTest(unittest.TestCase):
  def test_flayer(self):
    in_layer = layers.InputLayer(shape=(None, 1, 100, 100))
    c = conv(filter_size=(3, 3))(nonlinearity=nonlinearities.linear)(pad='same')(num_filters=32)
    assert not isinstance(c, layers.Layer)

    conv32 = c(in_layer)
    conv64 = c(num_filters=64)(in_layer)
    assert isinstance(conv32, layers.Conv2DLayer) and conv32.num_filters == 32
    assert isinstance(conv64, layers.Conv2DLayer) and conv64.num_filters == 64

    net = subnetworks.chain(
      layers=conv(filter_size=(3, 3)),
      num_filters=[32, 64, 128]
    )(incoming = in_layer)

    from craynn.utils.visualize import draw_to_file
    draw_to_file(layers.get_all_layers(net), 'test_net.png')

if __name__ == '__main__':
  unittest.main()
