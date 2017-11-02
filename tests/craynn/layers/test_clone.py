import numpy as np
import theano
theano.config.floatX = 'float32'

import theano.tensor as T

from craynn.layers import *
from craynn.subnetworks import *
from craynn.networks import *

from lasagne.layers import get_output_shape

def test_clone():
  clone_op = clone(conv(13))

  net1 = net((None, 3, 16, 16))(
    clone_op
  )

  net2 = net((None, 3, 17, 17))(
    clone_op
  )

  assert get_output_shape(net1.outputs[0]) == (None, 13, 14, 14), get_output_shape(net1.outputs[0])
  assert get_output_shape(net2.outputs[0]) == (None, 13, 15, 15), get_output_shape(net2.outputs[0])

  X = T.ftensor4()

  predict1 = theano.function([X], net1(X)[0])
  predict2 = theano.function([X], net2(X)[0])

  for _ in range(16):
    X = np.random.uniform(size=(15, 3, 17, 17)).astype('float32')
    X_ = X[:, :, :-1, :-1]
    assert predict1(X_).shape == (15, 13, 14, 14), predict1(X_).shape
    assert predict2(X).shape == (15, 13, 15, 15), predict2(X).shape

    assert np.allclose(predict1(X_), predict2(X)[:, :, :-1, :-1], atol=1.0e-6), np.max(np.abs(predict1(X_)- predict2(X)[:, :, :-1, :-1]))