from craynn.layers import *
from craynn.subnetworks import *
from craynn.networks import *

def test_rconv():
  discriminator = net((None, 1, 28, 28))(
    rconv(64), rconv(64), max_pool(),
    rconv(96), rconv(96), max_pool(),
    rconv(128), flatten(),
    dense(1, f=lambda x: x), flatten(outdim=1)
  )

  kernels = get_kernels_by_type(discriminator.layers, 'restricted')
  assert len(kernels) == 5