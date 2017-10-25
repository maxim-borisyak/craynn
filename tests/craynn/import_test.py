import unittest

import numpy as np

import theano
theano.config.floatX = 'float32'
import theano.tensor as T

import craynn

from lasagne import layers

from craynn.layers import *
from craynn.subnetworks import *
from craynn.networks import *

from craynn.layers.random import GaussianInput
from craynn.viz import draw_to_file

class ImportTest(unittest.TestCase):
  def test_import(self):
    test_input = layers.InputLayer(shape=(None, 1, 28, 28))
    random_input = GaussianInput(shape_or_layer=test_input)

    nn = net(test_input, random_input)(
      [
        (lambda inputs: inputs[0], conv(8), conv(16)),
        (lambda inputs: inputs[1], conv(23), conv(16)),
      ],
      maximum()
    )

    print(nn.description())

    draw_to_file(layers.get_all_layers(nn.outputs), 'test.png')

    X = T.ftensor4()
    y, = nn(X)

    f = theano.function([X], y)

    test_input = np.random.uniform(size=(13, 1, 28, 28)).astype('float32')
    output = f(test_input)

    assert output.shape == (13, 16, 24, 24)


if __name__ == '__main__':
  unittest.main()
