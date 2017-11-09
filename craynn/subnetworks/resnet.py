from ..layers import *
from .conv_nets import double_diff
from .common import achain

__all__ = [
  'residual_connection',
  'res_block',
  'bottleneck_res_block'
]

def _residual_connection(incoming, conv_op, elementwise_sum_op=elementwise_sum()):
  net = conv_op(incoming)
  return elementwise_sum_op([incoming, net])

residual_connection = lambda op, elementwise_sum_op=elementwise_sum(): lambda incoming:\
  _residual_connection(incoming, op, elementwise_sum_op)

res_block = lambda num_filters, conv=double_diff, elementwise_sum_op=elementwise_sum(): lambda incoming:\
  _residual_connection(incoming, conv(num_filters), elementwise_sum_op)

bottleneck_res_block = lambda num_filters, factor=4, conv=diff, conv1x1=diff1x1, f=None, elementwise_sum_op=elementwise_sum(): lambda incoming: \
  _residual_connection(
    incoming,
    conv_op = achain(
      conv1x1(num_filters // factor, f=f),
      conv(num_filters // factor, f=f),
      conv1x1(num_filters, f=f)
    ),
    elementwise_sum_op=elementwise_sum_op
  )

