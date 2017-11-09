import numpy as np

import theano
import theano.tensor as T

from lasagne.updates import get_or_compute_grads, total_norm_constraint

import lasagne

from collections import OrderedDict

__all__ = [
  'clipped_gradients',
  'careful',
  'normalized_gradients',
  'careful_rmsprop',
  'careful_adamax',
  'careful_adam'
]

def clipped_gradients(updates, grad_clipping=1.0, epsilon=1.0e-6):
  """
  Clips gradients before passing them to updates.

  :param updates: any `lasagne.updates` optimizer or similar.
  :param grad_clipping: maximal norm of the gradients.
  :param epsilon:
  :return:
  """
  def u(loss_or_grads, params, *args, **kwargs):
    grads = get_or_compute_grads(loss_or_grads, params)
    grads = total_norm_constraint(grads, max_norm=grad_clipping, epsilon=epsilon)

    return updates(grads, params, *args, **kwargs)
  return u

careful = clipped_gradients

def normalized_gradients(updates, epsilon=1.0e-6):
  def u(loss_or_grads, params, *args, **kwargs):
    grads = get_or_compute_grads(loss_or_grads, params)
    gnorm = T.sqrt(sum(T.sum(g ** 2) for g in grads) + epsilon)
    grads = [g / gnorm for g in grads]

    return updates(grads, params, *args, **kwargs)
  return u

def careful_rmsprop(loss_or_grads, params,  grad_clipping=1.0, learning_rate=1.0, rho=0.9, epsilon=1e-6):
  """
  RMSProp with gradient clipping. Shortcut for `clipped_gradients(rmsprop)`.
  """
  return clipped_gradients(lasagne.updates.rmsprop, grad_clipping, epsilon)(
    loss_or_grads, params,
    learning_rate=learning_rate, rho=rho, epsilon=epsilon
  )

def careful_adam(loss_or_grads, params, grad_clipping=1.0, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
  """
  Adam with gradient clipping. Shortcut for `clipped_gradients(adam)`.
  """
  return clipped_gradients(lasagne.updates.adam, grad_clipping, epsilon)(
    loss_or_grads, params,
    learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon
  )

def careful_adamax(loss_or_grads, params, grad_clipping=1.0, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
  """
  Adamax with gradient clipping. Shortcut for `clipped_gradients(adamax)`.
  """
  return clipped_gradients(lasagne.updates.adamax, grad_clipping, epsilon)(
    loss_or_grads, params,
    learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon
  )

def cruel_rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6,
                  grad_clipping=1.0e-2, param_clipping=1.0e-2):
  """
  A version of careful RMSProp for Wassershtein GAN. 
  :param epsilon: small number for computational stability.
  :param grad_clipping: maximal norm of gradient, if norm of the actual gradient exceeds this values it is rescaled.
  :param param_clipping: after each update all params are clipped to [-`param_clipping`, `param_clipping`].
  :return: 
  """
  grads = get_or_compute_grads(loss_or_grads, params)
  updates = OrderedDict()
  grads = total_norm_constraint(grads, max_norm=grad_clipping, epsilon=epsilon)

  # Using theano constant to prevent upcasting of float32
  one = T.constant(1)

  for param, grad in zip(params, grads):
    value = param.get_value(borrow=True)
    accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                         broadcastable=param.broadcastable)
    accu_new = rho * accu + (one - rho) * grad ** 2
    updates[accu] = accu_new

    updated = param - (learning_rate * grad / T.sqrt(accu_new + epsilon))

    if param_clipping is not None:
      updates[param] = T.clip(updated, -param_clipping, param_clipping)
    else:
      updates[param] = updated

  return updates