"""
CRAYNN is just another layer of abstraction for neural networks.

Build on top of lasagne (which in turn is build on theano),
it focuses on macro-architecture.

Important note: this package is poorly designed, unstable and lacks documentation.
"""

import os as _os
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from . import layers