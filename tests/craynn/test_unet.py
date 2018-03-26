import craynn
from craynn.viz import draw_to_file
from craynn.layers import *
from craynn.subnetworks import *
from craynn.networks import *

from lasagne import *

def test_net():
  nn = net((None, 1, 28, 28))(
    unet(
      down_ops=[
        diff(16),
        achain(max_pool(), diff(32)),
        achain(max_pool(), diff(64))
      ],
      middle_ops=[
        diff(33),
      ],
      up_ops=[
        achain(diff(32), upscale(),),
        achain(diff(16), upscale(), ),
        diff(1)
      ]
    )
  )


  draw_to_file(nn, 'test.png')
