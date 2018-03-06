"""
General rules held in this module:
- params ending with `_op` like `conv_op` or simply op require function layer -> layer, e.g. conv(128, f=...)
- params not ending with `_op` like `conv` require function with similar signature as
    layer/subnetwork with the same name. Such parameters allow to override layer/subnetwork used.
    For example, overriding default nonlinearity: conv=lambda num_filters: conv(num_filters, f=T.nnet.sigmoid).
- usually, module provides two similar implementations of the same subnetwork like
  fire and fire_block. Longer name = more flexibility but more parameters.
"""

from .conv_nets import *
from .resnet import *
from .meta import *
from .fire_nets import *
from .scale import *
from .prop_nets import *
from .u_nets import *
from .column_nets import *
from .energy_based import *
from .maxout_nets import *
