import tensorflow as tf
from .common import get_grads

__all__ = [
  'sgd'
]

def sgd(loss_or_grads, params, learning_rate):
  """
  Stochastic Gradient Descent.
  Actually, stochastic or not depends on you.

  :return: grouped assign ops.
  """
  grads = get_grads(loss_or_grads, params)
  updates = list()

  for param, grad in zip(params, grads):
    updates.append(
      tf.assign_sub(param, learning_rate * grad)
    )

  return tf.group(*updates)