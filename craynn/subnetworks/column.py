from ..layers import *

__all__ = [
  'column_module', 'column'
]

def _column_module(incoming, ops, channel_pool=channel_rpool(2), concat=concat()):
  return concat([
    op(channel_pool(incoming))
    for op in ops
  ])

column_module = lambda ops, channel_pool=channel_rpool(2), concat=concat(): lambda incoming: \
  _column_module(incoming, ops, channel_pool, concat)

column = lambda ops: lambda incoming: \
  _column_module(incoming, ops, channel_rpool(len(ops)), concat)