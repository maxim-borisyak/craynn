"""
CRAYNN is just another layer of abstraction for neural networks.

Build on top of lasagne (which in turn is build on theano),
it focuses on macro-architecture.

Important note: this package is poorly designed, unstable and lacks documentation.
"""

from . import init
from . import layers
from . import networks
from . import nonlinearities
from . import subnetworks
from . import updates
from . import utils
from . import viz
