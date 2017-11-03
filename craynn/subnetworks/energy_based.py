import theano.tensor as T

from ..layers import *

__all__ = [
  'mse_energy'
]

def _mse_energy(incoming, op, input_preprocessing_op, output_preprocessing_op):
  return global_pool(f=T.sum)(
    elementwise(lambda a, b: (a - b) ** 2)([
      input_preprocessing_op(incoming),
      output_preprocessing_op(op(incoming))
    ])
  )

mse_energy = lambda op, input_preprocessing_op=nothing, output_preprocessing_op=nothing: lambda incoming: \
  _mse_energy(incoming, op, input_preprocessing_op, output_preprocessing_op)