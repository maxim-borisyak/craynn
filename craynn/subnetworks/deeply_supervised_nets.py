from .meta import achain

__all__ = [
  'deeply_supervised'
]

def _deeply_supervised(incoming, *ops, pool_op):
  net = incoming
  companions = list()

  for op in ops:
    net = achain(op)(net)
    companions.append(pool_op(net))

  return companions

deeply_supervised = lambda pool_op: lambda *ops: lambda incoming: _deeply_supervised(incoming, *ops, pool_op=pool_op)

