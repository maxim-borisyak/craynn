from ..layers import *

def _mse_energy(incoming, op):
  return elementwise(lambda a, b: (a - b) ** 2)([
    incoming,
    op(incoming)
  ])

mse_energy = lambda op: lambda incoming: \
  _mse_energy(incoming, op)