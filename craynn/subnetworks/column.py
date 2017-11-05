from ..layers import *

__all__ = [
  'column_module', 'column'
]

def _column_module(incoming, ops, factor_pool_op=channel_factor_pool(2), concat_op=concat()):
  return concat_op([
    op(factor_pool_op(incoming))
    for op in ops
  ])

column_module = lambda ops, factor_pool_op=channel_factor_pool(2), concat_op=concat(): lambda incoming: \
  _column_module(incoming, ops, factor_pool_op, concat_op)

column = lambda num_filters, depth, conv=conv, factor_pool=channel_factor_pool: lambda incoming: \
  _column_module(
    incoming,
    [conv(num_filters // depth) for _ in range(depth)],
    channel_factor_pool(depth),
    concat()
  )