### this file was shamefully derived from https://github.com/amitgroup/amitgroup/tree/master/amitgroup

from .utils import onehot

"""
Copyright (c) 2010-2014 Benjamin Peterson

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from array import array
import struct

ROOT_URL = 'http://yann.lecun.com/exdb/mnist/'

TRAIN_DATA = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'

TEST_DATA = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

def download_and_save(root, file):
  import os

  path = os.path.join(root, file)
  if os.path.exists(path):
    raise IOError('Path %s already exists!' % path)

  import urllib.request

  import warnings
  warnings.warn('Downloading %s'% (ROOT_URL + file))

  response = urllib.request.urlopen(ROOT_URL + file)
  data = response.read()  # a `bytes` object

  with open(path, 'wb') as f:
    f.write(data)

  return path

def get(root, file):
  import gzip
  import os

  try:
    path = download_and_save(root, file)
  except IOError:
    path = os.path.join(root, file)

  with gzip.open(path, 'rb') as f:
    return f.read()


def mnist(root='./mnist', one_hot=False, cast=None):
  """
  Looks for MNIST archive in `root`, if not found then downloads it.

  :return: X_train, y_train, X_test, y_test
  """

  import os
  try:
    os.makedirs(root)
  except:
    pass

  train_labels_raw = get(root, TRAIN_LABELS)
  y_train = np.array(array("b", train_labels_raw[8:]), dtype='uint8')

  test_labels_raw = get(root, TEST_LABELS)
  y_test = np.array(array("b", test_labels_raw[8:]), dtype='uint8')

  train_images_raw = get(root, TRAIN_DATA)
  _, _, rows, cols = struct.unpack(">IIII", train_images_raw[:16])
  X_train = np.array(array("b", train_images_raw[16:]), dtype='uint8').reshape(-1, 1, rows, cols)

  test_images_raw = get(root, TEST_DATA)
  _, _, rows, cols = struct.unpack(">IIII", test_images_raw[:16])
  X_test = np.array(array("b", test_images_raw[16:]), dtype='uint8').reshape(-1, 1, rows, cols)

  if one_hot:
    y_train = onehot(y_train)
    y_test = onehot(y_test)

  if cast is True:
    cast = 'float32'

  if cast is not None:
    X_train = X_train.astype(cast)
    X_test = X_test.astype(cast)
    y_train = y_train.astype(cast)
    y_test = y_test.astype(cast)

  return X_train, y_train, X_test, y_test