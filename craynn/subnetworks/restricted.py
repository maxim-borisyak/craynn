import theano
import theano.tensor as T
from .common import get_kernels_by_type
from ..utils import lsum

def restricted_kernel_stabilizer(net, C=1.0, type='quadratic'):
  kernels = get_kernels_by_type(net, 'restricted')

  if type == 'quadratic':
    reg = lambda W: (T.sqrt(T.sum(W ** 2)) - C) ** 2
  elif type == 'linear':
    reg = lambda W: abs(T.sum(W ** 2) - C)
  else:
    raise ValueError('%s is not a proper type of stabilizer. Possible values: ["quadratic", "linear"]')

  if len(kernels) == 0:
    return T.constant(0.0)
  else:
    return lsum([ reg(W) for W in kernels ])