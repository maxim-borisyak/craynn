import unittest


class ImportTest(unittest.TestCase):
  def test_import(self):
    from craynn import utils as nnutils
    from craynn import layers as clayers
    from craynn import subnetworks
    from craynn import networks


if __name__ == '__main__':
  unittest.main()
