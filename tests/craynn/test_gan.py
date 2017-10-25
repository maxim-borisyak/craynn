import unittest

import theano
import theano.tensor as T
from craynn.layers import *
from craynn.subnetworks import *

from craynn.networks import StageGAN, Net
from craynn.objectives import cross_entropy

class TestGAN(unittest.TestCase):
  def test_import(self):
    pass

if __name__ == '__main__':
  unittest.main()
