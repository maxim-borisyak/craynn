"""
CRAYNN is just another layer of abstraction for neural networks.

Build on top of lasagne (which in turn is build on theano),
it focuses on macro-architecture.

Important note: this package is poorly designed, unstable and lacks documentation.
"""

from . import layers
from . import init
from . import nonlinearities
