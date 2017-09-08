import unittest


class ImportTest(unittest.TestCase):
  def test_import(self):
    from craynn import utils as nnutils
    from craynn import layers as clayers
    from craynn.subnetworks import *
    from craynn.networks import *


if __name__ == '__main__':
  unittest.main()
