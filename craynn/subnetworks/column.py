from ..layers import *

__all__ = [
  'column_module', 'column'
]

def _column_module(incoming, ops, factor_pool_op=channel_factor_pool(2), merge_op=elementwise_mean()):
  return merge_op([
    op(factor_pool_op(incoming))
    for op in ops
  ])

column_module = lambda ops, factor_pool_op=channel_factor_pool(2), merge_op=elementwise_mean(): lambda incoming: \
  _column_module(incoming, ops, factor_pool_op, merge_op)

column = lambda num_filters_per_column, column_channels, number_of_columns=2, conv=conv, factor_pool=channel_factor_pool, merge_op=elementwise_mean(): lambda incoming: \
  _column_module(
    incoming,
    [conv(num_filters_per_column) for _ in range(number_of_columns)],
    channel_pool(column_channels),
    merge_op
  )