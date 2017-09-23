import unittest

import theano
import theano.tensor as T

from lasagne import *
from craynn.layers import *
from craynn.subnetworks import *

from craynn.networks import StageGAN, Net
from craynn.objectives import cross_entropy

from craynn.viz import draw_to_file

class TestChains(unittest.TestCase):
  def test_import(self):

    X = T.ftensor4()
    Z = T.ftensor4()
    generator = Net(conv(32), img_shape=(1, 128, 128))

    linear = lambda x: x
    get_cnn = lambda *fs: cnn(fs, conv_op=freeze, pool_op=floating_meanpool, companion=mean_companion(f=linear))
    bases = achain([
      get_cnn(),
      get_cnn(8, ),
      get_cnn(8, 16),
      get_cnn(8, 16, 32),
      get_cnn(8, 16, 32, 64),
    ])

    discriminator = Net(bases, img_shape=(1, 128, 128))

    gan = StageGAN(
      loss=lambda a, b: cross_entropy(a, b, mode='linear'),
      discriminators=discriminator,
      generator=generator
    )

    a, b = gan(X, Z)
    print a
    print b


if __name__ == '__main__':
  unittest.main()
